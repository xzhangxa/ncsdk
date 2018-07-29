# Copyright 2018 Intel Corporation.
# The source code, information and material ("Material") contained herein is
# owned by Intel Corporation or its suppliers or licensors, and title to such
# Material remains with Intel Corporation or its suppliers or licensors.
# The Material contains proprietary information of Intel or its suppliers and
# licensors. The Material is protected by worldwide copyright laws and treaty
# provisions.
# No part of the Material may be used, copied, reproduced, modified, published,
# uploaded, posted, transmitted, distributed or disclosed in any way without
# Intel's prior express written permission. No license under any patent,
# copyright or other intellectual property rights in the Material is granted to
# or conferred upon you, either expressly, by implication, inducement, estoppel
# or otherwise.
# Any license under such intellectual property rights must be express and
# approved by Intel in writing.

import numpy as np
import re
import math
from collections import OrderedDict
from heapq import heappush, heappop
from mvnctools.Controllers.PingPong import PingPongSchedule, getManualHwSchedule, get_null_terminating_name
from mvnctools.Controllers.ProductionSchedulers import googlenet, yolo_tiny, vgg16
import mvnctools.Controllers.Globals as GLOBALS
from mvnctools.Models.EnumDeclarations import *
from mvnctools.Models.NetworkStage import NetworkStage
# This class represent an NCE module


def round_up(x, mult):
    return ((x + mult - 1) // mult) * mult

class NCE_Scheduler():
    def __init__(self, support_split_over_c = True, support_split_over_w = False, enable_permutation = False):

        self.support_split_over_c = support_split_over_c
        self.support_split_over_w = support_split_over_w
        self.enable_permutation = enable_permutation

        # cycles of overhead for each split.
        self.split_by_k_overhead 	= 7000
        self.split_by_c_overhead 	= 30000
        self.split_by_w_overhead 	= 5000

        self.solution = {}
        self.split = {}
        self.ordered_dict = OrderedDict()

        self.bytesPerPixel = 2
        self.ddr2ddr = False
        self.forbidden_modes = []

        self.DPExCh = OrderedDict()
        for i in range(5):
            self.DPExCh['mode{}'.format(i)] = 2**i

    def getSolution(self, stage):
        if stage in self.solution.keys():
            return self.solution[stage]
        else:
            return self.opt_operation(stage)[0]

    def getSplits(self, stage):
        if stage in self.split.keys():
            return self.split[stage]
        else:
            return self.opt_operation(stage)[1][0]

    def scheduling_operations(self, net, verbose = False):
        '''
        Rescheduling operations, find best memory layout and splits/tiles
        write op configuration to PingPong file

        :param net: Network object
        :param verbose: verbose output
        :return:
        '''
        pingPongPair = getManualHwSchedule()

        self.permutationList = [i for i in range(len(net.stageslist))]

        stage_dict = OrderedDict((get_null_terminating_name(stage.name), idx) for (idx, stage) in enumerate(net.stageslist))
        subgraph_lst = self.extract_subgraphs(net)

        # optimize each subgraphs:
        subgraph_conf = []
        print("Network has {} subgraphs".format(len(subgraph_lst)))
        for idx, (stage_start, stage_end) in enumerate(subgraph_lst):
            print('\tOptimize subgraph {}'.format(idx+1))
            subgraph_start_loc_lst = ['D']
            for subgraph_start_loc in subgraph_start_loc_lst:
                self.process_subgraph(net.stageslist[stage_dict[stage_start]:stage_dict[stage_end]+1],
                                net, stage_dict[stage_start],stage_dict[stage_end]+1, subgraph_start_loc)

                subgraph_conf.append({subgraph_start_loc:{'D':0}})

        linear_network = net.stageslist.copy()
        for idx in reversed(range(len(subgraph_conf))):
            (stage_start, stage_end) = subgraph_lst[idx]
            linear_network[stage_dict[stage_start]+1:stage_dict[stage_end]+1] = [dict(subgraph_conf[idx])]

        self.process_branch(linear_network, final_config = 'D')

        if verbose:
            print('************************************************')
            if self.enable_permutation:
                print("Rescheduled operation order")
            for i in self.permutationList:
                stageName = get_null_terminating_name(net.stageslist[i].name)
                try:
                    print(stageName, self.ordered_dict[stageName])
                except Exception as e:
                    pass
            print('************************************************')

        # apply scheduling to ping-pong
        for i in self.permutationList:
            key = get_null_terminating_name(net.stageslist[i].name)
            if key in self.ordered_dict.keys():
                pingPongPair.ordered_dict[key] = self.ordered_dict[key]
        pingPongPair.enable_perm = self.enable_permutation

    def extract_subgraphs(self, net):
        '''
        Extract subgraphs from network
        :param net: Network object
        :return subgraph_lst: list of subgraphs in the format (start_elem, end_elem).
        element in a subgraph should be adjacent in
        '''

        def subbranch_length(stage):
            child_stage = stage
            subbranch_len = 1
            while not child_stage.concatResult:
                if (len(child_stage.tail) != 1):
                    return subbranch_len
                child_stage = child_stage.tail[0]
                subbranch_len += 1
            return subbranch_len

        def subgraph_border(stage):
            child = stage.tail[-1]
            while not child.concatResult:
                child = child.tail[0]
            return (get_null_terminating_name(stage.name), get_null_terminating_name(child.name))

        def subbranch_tail(stage):
            child_stage = stage
            while not child_stage.concatResult:
                if (len(child_stage.tail) != 1):
                    return set([tail_stage for tail_stage in net.stageslist if tail_stage.top != None and get_null_terminating_name(child_stage.name) in tail_stage.top])
                child_stage = child_stage.tail[0]
            return set([tail_stage for tail_stage in net.stageslist if tail_stage.top != None and get_null_terminating_name(child_stage.name) in tail_stage.top])

        def is_subgraph_valid(stage):
            # Only valid sungraph (the ones with max lengt = 2, i.e the only supported for now in the process subgraph routine)
            cond1 = max([subbranch_length(child) for child in stage.tail]) <= 2
            # The subgraph must merge together
            cond2 = len(list(set.union(*[subbranch_tail(child) for child in stage.tail]) - set.intersection(*[subbranch_tail(child) for child in stage.tail]))) == 0
            
            return cond1 and cond2


        # list of stages with more than one element in the tail
        subgraph_in_stages_names = list(set([stage.tail[0].top[-1] for stage in net.stageslist if len(stage.tail)>1]))
        subgraph_in_stages = [stage for stage in net.stageslist if get_null_terminating_name(stage.name) in subgraph_in_stages_names]

        valid_in_stages = [stage for stage in subgraph_in_stages if is_subgraph_valid(stage)]

        if (len(valid_in_stages) != len(subgraph_in_stages)):
            print("There are non valid subgraphs => optimize valid subgraphs and DDR->DDR other layers")
            self.ddr2ddr = True

        subgraph_lst = [subgraph_border(stage) for stage in valid_in_stages]

        return subgraph_lst

    def process_branch(self, stages, initial_config = 'D', final_config = 'D'):
        '''
        Process a branch in order to find best memory configuration
        :param stages: branch stages
        :param initial_config: branch input memory location (default DDR)
        :param final_config: branch output memory location (default DDR)
        :return:
        '''
        if len(stages) == 1:
            cost = self.calc_branch_cost(stages[0], initial_config, final_config)
            self.gen_ping_pong_pair(stages[0], final_config, initial_config)
            self.solution[get_null_terminating_name(stages[0].name)], self.split[get_null_terminating_name(stages[0].name)] = self.opt_operation(stages[0])
            return cost

        conf_lst = ['D', 'L', 'R']
        graph = OrderedDict()
        # Generate graph
        for idx, stage in enumerate(stages):
            if isinstance(stage, dict):
                stageName="subgraph_{}".format(idx)
            else:
                stageName = get_null_terminating_name(stage.name)
            if idx == 0:
                start_node = '{}_{}'.format(stageName, initial_config)
                elem = Vertex(start_node,initial_config)
                elem.distance = 0
                if len(stages) == 2:
                    next_conf_lst = [final_config]
                else:
                    next_conf_lst = conf_lst
                for next_conf in next_conf_lst:
                    if isinstance(stages[idx+1], dict):
                        nextStageName="subgraph_{}_{}".format(idx+1, next_conf)
                    else:
                        nextStageName = "{}_{}".format(get_null_terminating_name(stages[idx+1].name), next_conf)
                    cost = self.calc_branch_cost(stages[idx+1], initial_config, next_conf)
                    elem.addConnection(nextStageName, cost, None)
                graph[start_node] = elem

            elif idx == len(stages)-1: #last
                end_node = '{}_{}'.format(stageName, final_config)
                graph[end_node] = Vertex(end_node, final_config)

            else:
                for conf in conf_lst:
                    elem = Vertex('{}_{}'.format(stageName, conf), conf)
                    if idx == len(stages) - 2:
                        next_conf_lst = [final_config]
                    else:
                        next_conf_lst = conf_lst

                    for next_conf in next_conf_lst:
                        if isinstance(stages[idx+1], dict):
                            nextStageName="subgraph_{}_{}".format(idx+1, next_conf)
                        else:
                            nextStageName = "{}_{}".format(get_null_terminating_name(stages[idx+1].name), next_conf)
                        cost = self.calc_branch_cost(stages[idx+1], conf, next_conf)
                        elem.addConnection(nextStageName, cost, None)
                    graph['{}_{}'.format(stageName, conf)] = elem

        opt_path, opt_cost = self.dijkstra(graph, start_node , [end_node])

        for idx, stage in enumerate(stages):
            if not isinstance(stage, dict):
                if idx == 0:
                    self.gen_ping_pong_pair(stage, opt_path[end_node][idx],initial_config)
                else:
                    self.gen_ping_pong_pair(stage, opt_path[end_node][idx],opt_path[end_node][idx-1])
                self.solution[get_null_terminating_name(stage.name)], self.split[get_null_terminating_name(stage.name)] = self.opt_operation(stage)

        return opt_cost[end_node]

    def calc_branch_cost(self, nextElem, currenConf, nextConf):
        '''
        Calculate the cost of a combination of in/out memoy location.
        The cost function is the amount of data transfer from/to DDR
        :param nextElem: N+1 operation
        :param currentConf: N memory configuration (DDR/CMX left or right)
        :param nextConf: N+1 memory configuration (DDR/CMX left or right)
        :return cost:
        '''
        if isinstance(nextElem,dict): # a subgraph can be represented by a dictionary
            try:
                return nextElem[currenConf][nextConf]
            except Exception as e:
                return np.inf

        # non hw layer must be DD
        if not self.isHWLayer(nextElem) and (currenConf != 'D' or nextConf != 'D'):
            return np.inf

        # non hw layers with split over input channel need streaming so they must be DD
        if (self.getSplits(nextElem) > 1) and (currenConf != 'D' or nextConf != 'D'):
            return np.inf

        # if an element has more than one stage in the tail, it should outputs in DDR
        if (len(nextElem.tail) > 1) and (nextConf != 'D'):
            return np.inf

        # if ddr2ddr flag is set, only this configuration is allowed
        if (self.ddr2ddr) and (currenConf != 'D' or nextConf != 'D'):
            return np.inf

        cost = 0
        if currenConf == 'D':
            cost += self.get_input_descriptors_size(nextElem)
            if nextConf == 'D':
                cost += self.get_descriptors_size(nextElem)
            elif nextConf == 'R':
                cost += np.inf
            else:
                if self.get_descriptors_size(nextElem) > GLOBALS.CMX_SIZE//2: #TODO: check this!
                    cost += np.inf

        if currenConf == 'L':
            if nextConf == 'D':
                cost += self.get_descriptors_size(nextElem)
            elif nextConf == 'L':
                cost += np.inf
            else:
                if (self.get_descriptors_size(nextElem) > GLOBALS.CMX_SIZE//2) or (self.get_input_descriptors_size(nextElem) > GLOBALS.CMX_SIZE//2):
                    cost += np.inf

        if currenConf == 'R':
            if nextConf == 'D':
                cost += self.get_descriptors_size(nextElem)
            elif nextConf == 'R':
                cost += np.inf
            else:
                if (self.get_descriptors_size(nextElem) > GLOBALS.CMX_SIZE//2) or (self.get_input_descriptors_size(nextElem) > GLOBALS.CMX_SIZE//2):
                    cost += np.inf
        return cost


    def process_subgraph(self, stages, net, start_idx, end_idx, subgraph_start_loc = 'D'):
        '''
        Process a subgraph in order to get optimal memory configuration for each element
        it is possible to fit the subgraph into CMX? is it if :
            a) no operation need streaming, each descriptor is smaller than CMX/2
            b) the sum of all the outputs before concat is lower than CMX
            c) available memory for each branch considering already allocated output
            d) branch with depth > 2 not supported (yet)
        :param stages: stages in the subgraph
        :param net: Network object
        :param start_idx: stages[0] index in net.stageslist
        :param end_idx: stages[-1] index in net.stageslist
        :param subgraph_start_loc: subgraph input memory location (default DDR)
        :return:
        '''
        in_cmx = True
        concat_size = 0
        branch_descriptor = []
        for stage in stages[0].tail:
            first_stage = stage
            max_descriptor = 0
            last_descriptor = 0
            branch_depth = 0
            while True:
                descriptor_size = self.get_descriptors_size(stage)
                max_descriptor = max(max_descriptor, last_descriptor + descriptor_size)
                last_descriptor = descriptor_size
                branch_depth += 1
                # print('\t\t{} descriptor size: {}'.format(get_null_terminating_name(stage.name), descriptor_size))
                if descriptor_size > GLOBALS.CMX_SIZE//2 or branch_depth > 2:
                    in_cmx = False 		# this operation need streaming
                if ((len(stage.tail) == 0) or len(stage.tail[0].top) > 1):
                    concat_size += descriptor_size
                    branch_descriptor.append((max_descriptor, descriptor_size, first_stage, stage))
                    break
                else:
                    stage = stage.tail[0]

        if self.enable_permutation:
            # generate permutation index
            sort_index = [i[0] for i in sorted(enumerate(branch_descriptor), key=lambda x:x[0], reverse=True)]
            stages_index = [0]
            for stage in [stages[0].tail[i] for i in sort_index]:
                while stage:
                    stages_index.extend([i for i, elem in enumerate(stages) if stage.name == elem.name])
                    if len(stage.tail) < 1:
                        break
                    stage = stage.tail[0]
            self.permutationList[start_idx:end_idx] = [ i + start_idx for i in stages_index]
            reordered_branch = sorted(branch_descriptor, key=lambda x:x[0], reverse=True)
        else:
            reordered_branch = branch_descriptor

        in_cmx = in_cmx and (concat_size < GLOBALS.CMX_SIZE)

        # Check if I have memory for branch allocation
        if in_cmx:
            already_allocated = 0
            for max_descriptor, descriptor_size, _, _ in reordered_branch:
                if already_allocated + max_descriptor < GLOBALS.CMX_SIZE:
                    already_allocated += descriptor_size
                else:
                    in_cmx = False

        if not in_cmx:
            print('\t\x1b[6;30;41m' + 'subgraph cannot be done in CMX'  + '\x1b[0m')
            for stage in stages[1:]:
                if stage in stages[0].tail: # Here enforce the fact that inceptions should start in DDR!
                    solution = self.gen_ping_pong_pair(stage, desc_location='D', previous_loc=subgraph_start_loc)
                else:
                    solution = self.gen_ping_pong_pair(stage, desc_location='D', previous_loc='D')
                self.solution[get_null_terminating_name(stage.name)], self.split[get_null_terminating_name(stage.name)] = solution
        else:
            print('\t\x1b[6;30;42m' + 'subgraph can be done in CMX'  + '\x1b[0m')
            # now I need to allocate each branch
            already_allocated = 0
            stage_layout_dict = {}
            for max_descriptor, descriptor_size, first_stage, last_stage in reordered_branch:
                branch_len = 1 # get branch length
                stage = first_stage
                while stage != last_stage:
                    branch_len += 1
                    stage = stage.tail[0]

                if branch_len % 2: # even ping pong: fist should be on right
                    current_slice = 'L'
                else:
                    current_slice = 'R'
                stage = first_stage
                stage_layout_dict[stage] = current_slice
                while stage != last_stage:
                    stage = stage.tail[0]
                    if current_slice == 'L':
                        current_slice = 'R'
                    else:
                        current_slice = 'L'
                    stage_layout_dict[stage] = current_slice

            if self.enable_permutation:
                stage_layout_dict[net.stageslist[self.permutationList[end_idx-1]]] += 'U'
            else:
                stage_layout_dict[net.stageslist[end_idx-1]] += 'U'

            for stage in stages[1:]:
                if stage in stages[0].tail: # Here enforce the fact that inceptions should start in DDR!
                    solution = self.gen_ping_pong_pair(stage, desc_location=stage_layout_dict[stage], previous_loc=subgraph_start_loc)
                else:
                    previous = [x for x in stages if get_null_terminating_name(x.name) == stage.top[0]]
                    solution = self.gen_ping_pong_pair(stage, desc_location=stage_layout_dict[stage], previous_loc=stage_layout_dict[previous[0]])
                self.solution[get_null_terminating_name(stage.name)], self.split[get_null_terminating_name(stage.name)] = solution


    def gen_ping_pong_pair(self, stage, desc_location, previous_loc):
        '''
        Generate a pair of key + tuple
            key is the layer name
            the tuple is in the format ('Streaming/split', 'In/Out location', 'InOut layoud')
                SS -> Enable streaming and split over height if necessary.
                SX -> Enable streaming, but don't split over height.
                X? -> Don't stream, and don't split over height. '?' is "don't care".
                D = DDR
                L/R = left right position in CMX
                U = Unload CMX, i.e. return data from CMX to DDR (blocking DMA)
        It also generate optimal stage solution

        :param stage: NetworkStage object
        :param desc_location: output location in memory (DDR, CMX left right)
        :param previous_loc: input location in memory (DDR, CMX left right)
        :return solution: stage optimal solution in the form (inCh, OutCh, [(oCh, mode), ...]), inCh splits
        '''

        if get_null_terminating_name(stage.name) == 'data':
            #TODO: maybe not required??
            # self.ordered_dict[get_null_terminating_name(stage.name)] = ('SS', 'DD', 'PP', 0)
            return None, 1

        if stage.top != None:
            layout = 'II'
        else:
            layout = 'PI'

        name = get_null_terminating_name(stage.name)

        if desc_location == 'C':
            if previous_loc == 'D':
                desc_location = 'L'
            elif previous_loc == 'L':
                desc_location = 'R'
            elif previous_loc == 'R':
                desc_location = 'L'
            else:
                raise ValueError("Unsupported data location {}".format(previous_loc))

        location = "{}{}".format(previous_loc, desc_location)

        # TODO: Implement me!
        reuse = 0
        streamingConf=('L', 256)

        solution = (self.getSolution(stage) , self.getSplits(stage))

        streaming = 'XX'
        # Streaming only for hw layer
        if self.isHWLayer(stage):
            # if input and output are in DDR you should select streaming
            if (location.lower() == 'dd'):
                # you need to split over h?
                if stage.op != StageType.fully_connected_layer:
                    streaming = 'SS'
                else:
                    streaming = 'SX'

        self.ordered_dict[name] = (streaming, location, layout, reuse, streamingConf)

        return solution

    def opt_operation(self, stage, verbose = False):
        '''
        Optimaze a single operation

        :param stage: NetworkStage object
        :param verbose: generate svg graph file
        :return solution: stage optimal solution in the form (inCh, OutCh, [(oCh, mode), ...]), inCh splits
        '''
        # op is the operation to run on the NCE
        if stage.op == StageType.convolution:
            return self.optimize_convolution(stage, verbose)
        elif stage.op == StageType.max_pooling or stage.op == StageType.average_pooling:
            return self.optimize_pooling(stage)
        elif stage.op == StageType.fully_connected_layer:
            # Split in the best mode possible. However, if there are more than one tiles, we force
            #   mode 0 and split again, because of a hardware bug (which does not reset the accumulator)
            solution = self.optimize_fc(stage)
            if len(solution[0][2]) > 1:
                solution = self.optimize_fc(stage,[0])
            return solution
        else:
            return [],[1]

    def optimize_convolution(self, stage, verbose=False):
        '''
        Optimize convolution

        :param stage: NetworkStage object
        :param verbose: generate svg graph file
        :return solution: stage optimal solution in the form (inCh, OutCh, [(oCh, mode), ...]), inCh splits
        '''
        if isinstance(stage, dict):
            param = stage  # debug
        else:
            param = {'k': stage.radixX, 's': stage.strideX,
                     'ic': stage.inputDimZ, 'oc': stage.outputDimZ,
                     'is': [stage.inputDimX, stage.inputDimY],
                     'os': [stage.outputDimX, stage.outputDimY]}

        if verbose:
            from mvnctools.Views.Graphs import viz_opt_graph

        graph = {}
        # Create graph with start and end node
        start_name = "{}".format(param['oc'])
        end_name = '0'

        graph[start_name] = Vertex(start_name)
        graph[start_name].distance = 0

        # graph[end_name] = Vertex(end_name)
        # graph[end_name].distance = np.inf

        if verbose:
            print('Input Channels {}, Output Channels {}'.format(param['ic'], param['oc']))
            print('Input Shape {}, Output Shape {}'.format(param['is'], param['os']))

        # heap = (cost, step, name, param, conf)
        heap, seen = [(0, 0, start_name, [], param)], set()
        iterations = 0

        # get valid modes: output descriptor should be less than 255!!
        valid_modes = self.get_valid_modes(param)

        while heap:
            # get the vertex with min distance
            try:
                (min_dis, min_step, min_name, min_conf, min_param) = heappop(heap)
            except Exception as e:
                for elem in heap:
                    print(elem)

                print(min_name)
                raise e


            if min_name not in seen:
                seen.add(min_name)

                # check end condition
                if min_name == end_name:
                    break

                iterations += 1

                # Generate neighbors, update their cost,
                # try to implement each mode
                for mode in valid_modes:

                    (op_cost, node_name, newparam, mode_conf, step, last) =  self.split_by_k(min_name, min_param, mode, min_step)

                    if op_cost < np.inf:
                        # print('Node {}: generate {}'.format(min_name, mode))
                        # calculate new distance
                        distance = graph[min_name].distance + op_cost
                        if node_name in graph.keys():
                            if distance < graph[node_name].distance:
                                graph[node_name].distance = distance
                                graph[node_name].previous = min_name
                        else:
                            # create new node
                            graph[node_name] = Vertex(node_name)
                            graph[node_name].distance = distance
                            graph[node_name].previous = min_name

                        # update connection (for visualization only)
                        graph[min_name].addConnection(node_name, cost = op_cost, conf = mode_conf)

                        # if now in seen, push in the heap
                        if node_name not in seen:
                            heappush(heap, (distance, step, node_name, mode_conf, newparam))

                        # add connection with end node if it is the last
                        if last:
                            graph[node_name].addConnection(end_name, 0, [])
                            if distance < graph[end_name].distance:
                                graph[end_name].distance = distance
                                graph[end_name].previous = node_name
                                heappush(heap, (distance, step, end_name, [], newparam))
                if verbose:
                    viz_opt_graph(graph, 'GRAPH/graph_{}'.format(iterations), [min_name])


        # print("Algorithm iterations: {}".format(iterations))
        # generate a list of node

        param_sequence = []
        name_sequence = [start_name]
        node = end_name

        if not end_name in graph.keys():
            # No direct path
            print("***Operation not supported!!!***")
            return (param['ic'], param['oc'], []), 1

        while graph[node].previous:
            name_sequence.append(graph[node].name)
            param_sequence.append(graph[graph[node].previous].connection_conf[node])
            node = graph[node].previous

        param_sequence.reverse()
        if verbose:
            viz_opt_graph(graph, 'GRAPH/graph_{}'.format(iterations), name_sequence)

        # param_sequence = param_sequence[:-1]
        if len(param_sequence) == 0:
            print("***Operation not supported!!!***")
            return (param['ic'], param['oc'], []), 1

        max_mode = max([conf[1][1] for conf in param_sequence])

        # modeN requires multiple of 2**N blocks
        ramBlocks = 1 << max_mode # select the max mode

        # inC must be a multiple of 2**mode
        inC = ((param['ic'] + ramBlocks - 1) // ramBlocks) * ramBlocks
        # outC must be a multiple of 8
        outC = ((param['oc'] +  7) // 8) * 8

        in_split_mode = [conf[1][0] for conf in param_sequence]
        solution = [(conf[0], conf[1][1]) for conf in param_sequence]
        in_splits = [conf[1][2] for conf in param_sequence]

        if not ((len(solution) > 0) and (len(set(in_splits)) <= 1)): # check if in split are equal
            print("***Attention!!! Input split is different for each tile***")
            new_param = param.copy()
            self.support_split_over_c = False
            if min(in_splits) > 1:
                max_split = max(in_splits)
                new_param['ic'] = inC//max_split
                in_splits = [max_split]
                (_, outC, solution), _ = self.optimize_convolution(new_param)
            else:
                in_splits = [1]
                (inC, outC, solution), _ = self.optimize_convolution(new_param)
            self.support_split_over_c = True
            

        if inC % max(in_splits):
            return (inC, outC, []), in_splits

        correct, wrong_mode = self.check_actual_implementation(param, (inC//max(in_splits), outC, solution))

        if not correct:
            print("Error in hw operation optimization: remove mode {}".format(wrong_mode))
            self.forbidden_modes.append('mode{}'.format(wrong_mode))
            solution, splits = self.optimize_convolution(param)
            self.forbidden_modes = []
            return solution, splits

        return (inC//max(in_splits), outC, solution), in_splits

    def get_valid_modes(self, param):
        valid_modes = []
        max_mode = np.ceil(np.log2(param['ic']))
        for mode in self.DPExCh.keys():
            if (int(np.ceil(param['oc'] * self.DPExCh[mode] / GLOBALS.n_DPE)) < 255) and (int(mode[-1]) <= max_mode) and (mode not in self.forbidden_modes):
                valid_modes.append(mode)
        return valid_modes

    # return
    def split_by_k(self, current, conv_param, mode, idx):
        '''
        Implementation of split by output channel

        :param current: current configuration
        :param conv_param: operation parameter
        :mode: selected mode
        :mode idx: split over k index
        :return (split_cost, node_name, newparam, step, last):
        '''
        if idx == 0:
            overhead = 0
        else:
            overhead = self.split_by_k_overhead

        # inC must be a multiple of 2**mode
        param = conv_param.copy()
        ramBlocks = 1 << int(mode[-1])
        param['ic'] = ((param['ic'] + ramBlocks - 1) // ramBlocks) * ramBlocks

        need_split_by_k = self.check_output_channels_constraint(param, mode)
        need_split_by_w_or_c = self.check_min_lines_constraint(param,mode)
        need_split_by_c = (self.check_coeff_size_constraint(param, mode) or self.check_coeff_line_constraint(param, mode))

        # split-by-w
        if need_split_by_w_or_c:
            # I can solve this in two mode: I can split over w or I can split input channels
            conf = [(self.split_by_c(current, param, mode, idx)), (self.split_by_w(current, param, mode, idx))]
            (op_cost, mode_conf) = min(conf, key = lambda t: t[0]) # find min cost
        # check if conf violates hw on coefficient size on coeff store ram
        elif need_split_by_c:
            op_cost, mode_conf = self.split_by_c(current, param, mode, idx)
        else:
            # cost function per convolution: k*k*w*h*ic/DPEs
            op_cost = np.power(param['k'],2)*param['os'][0]*param['os'][1]*param['ic']/self.DPExCh[mode]
            mode_conf = ('k', int(mode[-1]), 1)
            # check input channel pr ram block constraint
            if (self.check_channels_per_ram_block(param, mode)):
                op_cost = np.inf
        # add overhead for higer mode
        op_cost += overhead
        # operation with permutaitons (ex: 1->2 2->1 should be connected to the same node)
        # node name is the amount of remaining output channels
        new_name = "{}".format(max(0, param['oc'] - int(GLOBALS.n_DPE/self.DPExCh[mode])))

        # add connection to new name, select only the last node
        # we need to split computation if output channel is greater than 256/DPExCh
        if (need_split_by_k):
            # we need to perform additional steps: update with the remaining output chanels
            new_param = conv_param.copy()
            new_param['oc'] = param['oc'] - int(GLOBALS.n_DPE/self.DPExCh[mode])
            return (op_cost, new_name, new_param, (int(GLOBALS.n_DPE/self.DPExCh[mode]), mode_conf), idx + 1, False)
        else:
            return (op_cost, new_name, param, (param['oc'], mode_conf), idx + 1, True)

    def split_by_w(self, current, param, mode, idx):
        '''
        Implementation of split by width

        :param current: current configuration
        :param param: operation parameter
        :mode: selected mode
        :mode idx: split over k index
        :return (split_cost, op_configuration):
        '''
        min_lines = param['k'] + param['s'] + 2
        # Space required by min lines = min_lines * input_width * input data type size * num input channels
        space_required = min_lines * 2 * param['is'][0] * param['ic']
        n_split_w = int(np.ceil(space_required/np.power(2, 17)))

        split_ow =  int(param['os'][0]/n_split_w)
        split_iw =  int(param['is'][0]/n_split_w)

        if (self.check_channels_per_ram_block(param, mode) or not self.support_split_over_w):
            return np.inf, ('w', int(mode[-1]), n_split_w)

        cost = 0
        for i in range(n_split_w):
            # calculate the split input and oputput w
            os_w = min(split_ow, param['os'][0]-i*split_ow)
            is_w = min(split_iw, param['is'][0]-i*split_iw)

            new_param = param.copy()
            new_param['os'] = os_w
            new_param['is'] = is_w # check for ic violation after the split
            if (self.check_coeff_size_constraint(new_param, mode) or self.check_coeff_line_constraint(new_param, mode)):
                return np.inf, []

            # cost = k*k*h*w / 2**mode + overhead
            cost += np.power(param['k'],2)*param['os'][0]*os_w*param['ic']/self.DPExCh[mode]  + self.split_by_w_overhead

        return cost, ('w', int(mode[-1]), n_split_w)

    def split_by_c(self, current, param, mode, idx):
        '''
        Implementation of split by input channel

        :param current: current configuration
        :param param: operation parameter
        :mode: selected mode
        :mode idx: split over k index
        :return (split_cost, op_configuration):
        '''
        # actual output channels
        actual_oc = min(param['oc'], int(GLOBALS.n_DPE/self.DPExCh[mode]))
        # Min lines  = Kernel_height + kernel stride + 2
        min_lines = param['k'] + param['s'] + 2
        # maximum ic that I can process without conflicting with min line constraint
        max_ic_minlines = np.floor(np.power(2,17)/(min_lines * self.bytesPerPixel * param['is'][0]))
        # maximum ic that I can process without conflicting with coefficient line per block constraint
        max_ic_ramblock = np.floor(256/(np.power(param['k'],2))*self.DPExCh[mode])

        # calculate the max input channels that can be processed without running out of memory:
        max_ic = int(min(max_ic_minlines, max_ic_ramblock))
        # calculate ramblock for the selected mode
        ramBlocks = 1 << int(mode[-1])
        while (max_ic > 0):
            # max_ic should be divisible by ramblocks and the split should be integer
            if (param['ic']%max_ic) or (max_ic%ramBlocks) :
                max_ic -= 1
            else:
                break
        if max_ic == 0:
            # This mode does not allows splits 
            return np.inf, ('c', int(mode[-1]), 0)

        # n of input split required (padded to next pow of 2)
        n_split_c = self.next_greater_power_of_2(param['ic']//max_ic)
        actual_ic_per_split = int(np.ceil(param['ic']/n_split_c))

        if(n_split_c < 2) or (actual_ic_per_split%ramBlocks) or not self.support_split_over_c:
            return np.inf, ('c', int(mode[-1]), 0)

        if (self.check_coeff_size_constraint({'ic':actual_ic_per_split, 'k':param['k'], 'oc':param['oc']}, mode) or
            self.check_channels_per_ram_block({'ic':actual_ic_per_split}, mode)  or 
            self.check_coeff_line_constraint({'ic':actual_ic_per_split, 'k':param['k']}, mode)):
            return np.inf, ('c', int(mode[-1]), n_split_c)

        cost = 0
        for i in range(n_split_c):
            # calculate the operation cost
            ic = min(actual_ic_per_split, param['ic']-i*actual_ic_per_split)
            cost += np.power(param['k'],2)*param['os'][0]*param['os'][1]*ic/self.DPExCh[mode]

        # add the cost of summation over input channels
        cost += (n_split_c-1)*(actual_oc*param['os'][0]*param['os'][1] + self.split_by_c_overhead)

        return cost, ('c', int(mode[-1]), n_split_c)


    def check_output_channels_constraint(self, param, mode):
        '''
        Check output channel constraint
        Computed output channels must be less than 256/2**mode

        :param param: operation parameter
        :mode: selected mode
        :return:
        '''
        return (param['oc'] > GLOBALS.n_DPE/self.DPExCh[mode])

    def check_coeff_size_constraint(self, param, mode):
        '''
        Check coefficient size constraint
        Coefficients size should be inside the 128 KB of the coefficient store RAM

        :param param: operation parameter
        :mode: selected mode
        :return:
        '''
        #
        actual_oc = min(param['oc'], GLOBALS.n_DPE/self.DPExCh[mode])
        coeff_size = np.power(param['k'],2)*param['ic']*round_up(actual_oc,8)*self.bytesPerPixel
        return (coeff_size > np.power(2, 17))

    def check_min_lines_constraint(self, param, mode):
        '''
        Check min lines constraint (before computation, how many lines need to be loaded initialy in the input RAM for ALL input channels)
        Min lines  = Kernel_height + kernel stride + 2

        :param param: operation parameter
        :mode: selected mode
        :return:
        '''
        min_lines = param['k'] + param['s'] + 2
        # Space required by min lines = min_lines * input_width (rounded to 16) * input data type size * num input channels
        space_required = min_lines * 2 * round_up(param['is'][0], 16) * param['ic']
        return (space_required > np.power(2, 17))

    def check_coeff_line_constraint(self, param, mode):
        '''
        Check coefficient lines per block constraint
        Channel per ram block : ram block is split along with DPE mode, and each rambclok gets an equal amount of input channels
        ex: mode 0 -> 1 ramblock, mode 1 -> 2 ramblock
        '''
        # Padded input channel
        ramblock = 1 << int(mode[-1])
        padded_in = ((param['ic'] + ramblock -1)//ramblock)*ramblock
        # Calculate ram blocks
        channel_per_ramblock = padded_in/self.DPExCh[mode]
        # coefficient number per line must be lower than 256
        check_coeff_line_per_block = param['k']*param['k']*channel_per_ramblock
        return (check_coeff_line_per_block > 256)

    def check_channels_per_ram_block(self, param, mode):
        '''
        Check channles per ram block constraint

        :param param: operation parameter
        :mode: selected mode
        :return:
        '''
        return (2**int(mode[-1]) > param['ic'])

    def check_actual_implementation(self, param, solution):
        '''
        Check if the actual implementation, with input/output padded
        breaks any hw constraint
        '''
        inC, outC, tiles = solution
        for k, mode in tiles:
            if self.check_output_channels_constraint({'oc':k}, 'mode{}'.format(mode)):
                return False, mode
            if self.check_coeff_size_constraint({'oc':k, 'k':param['k'], 'ic':inC}, 'mode{}'.format(mode)):
                return False, mode
            if self.check_min_lines_constraint({'k':param['k'], 's':param['s'], 'is':param['is'], 'ic':inC}, 'mode{}'.format(mode)):
                return False, mode
            if self.check_coeff_line_constraint({'k':param['k'], 'ic':inC}, 'mode{}'.format(mode)):
                return False, mode
            if self.check_channels_per_ram_block({'ic':inC}, 'mode{}'.format(mode)):
                return False, mode
        return True, None

    def next_greater_power_of_2(self, x):
        '''
        Find next power of two

        :param x:
        :return next greater power of two:
        '''
        return 2**(x-1).bit_length()

    def is_power2(self, x):
        '''
        CHeck if x is a power of two

        :param x:
        :return True if the number is a power of two:
        '''
        return x != 0 and ((x & (x - 1)) == 0)

    def optimize_pooling(self, stage):
        '''
        Optimize Overlapping Pooling layers

        :param stage: NetworkStage object
        :return solution: stage optimal solution in the form (inCh, OutCh, [(oCh, mode), ...]), inCh splits
        '''
        sub_layers = [(16, 4) for i in range(stage.inputDimZ // 16)]
        rem = stage.inputDimZ % 16
        if(rem != 0):
            sub_layers.append((round_up(rem, 16), 4))

        return (round_up(stage.inputDimZ, 16), round_up(stage.inputDimZ, 16), sub_layers), [1]


    def optimize_fc(self, stage, modes = [0, 1, 2, 3, 4]):
        '''
        Optimize FullyConnected layers

        :param stage: NetworkStage object
        :param modes: availabe mode list
        :return solution: stage optimal solution in the form (inCh, OutCh, [(oCh, mode), ...]), inCh splits
        '''
        inW = stage.inputDimZ
        inN = stage.outputDimZ

        solutions = []
        solution = None
        for mode in modes:
            ramblocks = 1 << mode
            maxW = ramblocks * 256
            maxN = 256 // ramblocks
            W = ((inW + ramblocks - 1) // ramblocks) * ramblocks
            N = ((inN + 7) // 8) * 8
            workW = min(W, maxW)
            workN = min(N, maxN)
            while workW >= ramblocks:
                countH = math.ceil(W / workW)
                countV = math.ceil(N / workN)
                cost = countH * countV * (workW // ramblocks + [0, 5, 11, 19, 31][mode])
                solutions.append((mode, W, N, workW, workN, countH, countV, cost))
                workW //= 2
        minCountH = min(solutions, key = lambda x:x[5])[5]
        zolutions = [sol for sol in solutions if sol[5] == minCountH]
        zolutions.sort(key = lambda x:x[7])
        if zolutions:
            mode, W, N, workW, maxN, countH, countV, cost = zolutions[0]
            tilesV = [[(workW, maxN, mode)] * countH] * countV
            solution = (W, N, tilesV)
        return solution, [1]

    def get_descriptors_size(self, stage):
        '''
        Get output descriptor size, with the input X dimension aligned to 16

        :param stage: NetworkStage
        :return output buffer size:
        '''
        outputBufferSize = stage.outputDimZ * stage.outputDimY * ((stage.outputDimX * self.bytesPerPixel + 15) // 16) * 16
        return outputBufferSize

    def get_input_descriptors_size(self, stage):
        '''
        Get input descriptor size, with the input X dimension aligned to 16

        :param stage: NetworkStage
        :return input buffer size:
        '''
        inputBufferSize = stage.inputDimZ * stage.inputDimY * ((stage.inputDimX * self.bytesPerPixel + 15) // 16) * 16
        return inputBufferSize

    def isHWLayer(self, stage):
        '''
        Return true if the stage can be inplemented in HW

        :param stage: NetworkStage
        :return:
        '''
        return (stage.op in [StageType.convolution, StageType.max_pooling, StageType.average_pooling, StageType.fully_connected_layer])

    def dijkstra(self, graph, start_node , end_nodes ):
        '''
        Implementation of Dijkstra's algorithm:  Dijkstra, E. W. (1959). "A note on two problems in connexion with graphs" (PDF). Numerische Mathematik. 1: 269–271. doi:10.1007/BF01386390.

        :param	graphs: a dictionary of connected vertex
        :param start_node: an input node
        :param end_nodes: a list of output node
        :return opt_path, opt_cost: optimal branch configuration and optimal cost for each output node (eg. output configuration)
        '''
        graph[start_node].distance = 0
        iterations = 0

        n_outputs = len(end_nodes)
        opt_cost = {}
        for min_name in end_nodes:
            opt_cost[min_name] = np.inf

        heap, seen = [(0, start_node)], set()

        while heap:

            # get the vertex with min distance
            (min_dis,min_name) = heappop(heap)

            # viz_opt_graph(graph, 'GRAPH/graph_{}'.format(iterations), [min_name])
            iterations += 1

            if min_name not in seen:
                seen.add(min_name)

                if min_name in end_nodes:
                    # print("Optimal Cost[{}]: {}".format(min_name, graph[min_name].distance))
                    opt_cost[min_name] = graph[min_name].distance
                    n_outputs -= 1
                    if (n_outputs == 0): break

                for neighbor in graph[min_name].connection.keys():
                    alt = graph[min_name].distance + graph[min_name].connection[neighbor]
                    if alt < graph[neighbor].distance:
                        #new shorter path found: update graph
                        graph[neighbor].distance = alt
                        graph[neighbor].previous = min_name

                    if neighbor not in seen:
                        heappush(heap, (graph[neighbor].distance, neighbor))


        # print("Optimal parameters recovered in {} iterations".format(iterations))
        opt_path = {}
        for end_node in end_nodes:
            if (opt_cost[end_node] < np.inf):
                # now reconstruct the path from the last node
                node = end_node
                param_sequence = []
                node_sequence = []
                while node:
                    param_sequence.append(graph[node].parameter)
                    node = graph[node].previous

                rev_param_sequence = []
                for elem in reversed(param_sequence):
                    rev_param_sequence.append(elem)
                opt_path[end_node] = rev_param_sequence # reversed: from first node to last

        # viz_opt_graph(graph, 'GRAPH/graph_{}'.format(iterations+1), rev_param_sequence)

        return opt_path, opt_cost


class Vertex():
    def __init__(self, name, parameter=None):
        self.visited = 0
        self.distance = np.inf
        self.parameter = parameter
        self.connection = OrderedDict()
        self.connection_conf = OrderedDict()
        self.name = name
        self.previous = None

    def addConnection(self, name, cost, conf):
        self.connection[name] = cost
        self.connection_conf[name] = conf

    def __str__(self):
        return '{}'.format(self.name)

    def __int__(self):
        return self.distance

    def __repr__(self):
        return str(self)

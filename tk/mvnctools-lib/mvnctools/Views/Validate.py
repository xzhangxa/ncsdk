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

from mvnctools.Models.EnumDeclarations import *
from mvnctools.Controllers.Metrics import *
from mvnctools.Controllers.EnumController import *


def top_test(result, exp_id, tolerance):
    """
    check if we pass the 'Top Test'. Whether that be a standard top-1, top-5, top-9 or another.
    :param result:
    :param exp_id:
    :param tolerance: amount of values to check
    :return: error code.
    """

    data = result.flatten()
    ordered = np.argsort(data)
    ordered = ordered[::-1]

    # This code block always seems to pass, so disabling now to resolve bug.
    # TODO: remove this block or reinstate if it was really useful
    # if len(result) < tolerance:
    #    print("Validation Pass")
    # return 0  # if our tolerance is higher than the amount of values, it
    # will definitely be correct

    # We use a set for future-proofing this check
    if int(exp_id) in ordered[:tolerance]:
        print("\nResult: Validation Pass\n")
        top = 0
    else:
        print("\nResult: Validation Fail\n")
        top = 1

    return top


def significant_classification_check(a, b, threshold, classification_type):

    significant_classifications = 0
    significant_matches = np.zeros((1))

    data = a.flatten()
    ordered = np.argsort(data)[::-1]

    for x in ordered:
        if data[x] > threshold:
            significant_classifications += 1

    ordered = ordered[:significant_classifications]

    data2 = b.flatten()
    ordered2 = np.argsort(data2)[::-1]
    ordered2 = ordered2[:significant_classifications]

    ordered_percentages = []
    for x in ordered[:significant_classifications]:
        ordered_percentages.append((data[x]))

    ordered_percentages2 = []
    for x in ordered2[:significant_classifications]:
        ordered_percentages2.append((data[x]))

    match_percentage = np.sum(ordered == ordered2)
    match_percentage /= ordered.flatten().shape[0]
    match_percentage *= 100

    match_percentage2 = np.in1d(ordered, ordered2)
    match_percentage2 = np.sum(match_percentage2)
    match_percentage2 /= ordered.flatten().shape[0]
    match_percentage2 *= 100

    test_status = "NO RESULT"

    if classification_type == ValidationStatistic.class_check_exact:
        test_status = "PASS" if np.all(match_percentage > 90) else "FAIL"
    if classification_type == ValidationStatistic.class_check_broad:
        test_status = "PASS" if np.all(match_percentage2 == 100) else "FAIL"

    print("------------------------------------------------------------")
    print(" Class Validation")
    print(" ------------------------------------------------------------")
    print(" Number of Significant Classifications: {}".format(
        significant_classifications))
    print(" Framework S-Classes:      {}".format(ordered))
    print(" Framework S-%:            {}".format(ordered_percentages))
    print(" Myriad S-Classifications: {}".format(ordered2))
    print(" Myriad S-%:               {}".format(ordered_percentages2))
    print(" Precise Ordering Match: {}%".format((match_percentage)))
    print(" Broad Ordering Match: {}%".format((match_percentage2)))
    print(" Result: {}".format(test_status))
    print("------------------------------------------------------------")


def top_classifications(values, amount=5):
    """
    print out the top X classifications from an array
    :param values: input array
    :param amount: how many to print
    :return: prints formatted to std output with no return values
    """
    data = values.flatten()
    ordered = np.argsort(data)
    ordered = ordered[::-1]  # Reverse to get a descending sort.
    for i, x in enumerate(ordered[:amount]):
        print(str(i + 1) + ')', x, data[x])

def validation_for_scaling(myriad_output, expected):
    """
    Compares NCS and Expected outputs and prints out the various metrics after
    scaling the layers as per the command line option --accuracy_adjust
    """
     # Accuracy calculations
    expected_values = expected.flatten()
    output_values = myriad_output.flatten().astype(np.float32)
    output_min = np.min(output_values)
    output_max = np.max(output_values)
    output_mean = np.mean(output_values, dtype=np.float32)
    output_std = np.std(output_values, dtype=np.float32)
    relative_error = np.absolute(expected_values - output_values) / np.maximum(np.absolute(expected_values), 0.00001)
    sum_relative_error = sum(relative_error)
    MRE = sum_relative_error / expected_values.size
    expected_squared = np.power(expected_values, 2)
    error_squared = np.power(expected_values - output_values, 2)
    expected_squared = np.maximum(expected_squared, 0.0001)
    print("max ", np.max(expected_squared), " min ", np.min(expected_squared))
    print("max ", np.max(error_squared), " min ", np.min(error_squared))
    # sqe = error_squared / expected_squared
    # print("max ", np.max(sqe), " min ", np.min(sqe))
    # MSQE2 = np.sum(sqe)
    MSE = np.sum(error_squared) / expected_values.size
    RMSE = np.sqrt(MSE)
    SNR = np.var(expected_values, dtype=np.float64) / np.var(expected_values - output_values, dtype=np.float64)
    SNR2 = np.sum(expected_squared, dtype=np.float64) / np.sum(error_squared, dtype=np.float64)
    print("MSE:{0:.6f}\tRMSE:{1:.6f}\tMRE:{2:.6f}\tSNR:{3:.6f}\tSNR2:{4:.6f}".format(MSE, RMSE, MRE, SNR, SNR2))
    print("min:{}\tmax:{}\tmean:{}\tstd:{}".format(output_min, output_max, output_mean, output_std))
    if expected.ndim == 3:
       print("--------------------- expected {}\n{}".format(expected.shape, expected[:4, :4, :4]))
       print("--------------------- actual {}\n{}".format(myriad_output.shape, myriad_output[:4, :4, :4]))
    elif expected.ndim == 1:
       print("--------------------- expected {}\n{}".format(expected.shape, expected_values[ :4]))
       print("--------------------- actual {}\n{}".format(myriad_output.shape, output_values[ :4]))

def validation(
        result,
        expected,
        expected_index,
        validation_type,
        filename,
        arguments):
    """
    Runs the validation step with the results and expected figures. Switches based on validation type
    :param result: resultant Matrix (from myriad)
    :param expected:  Expected Matrix (from NN program)
    :param expected_index: index that is predicted to come out.
    :param validation_type: enum
    :param filename: used to write csv for compare_matrices
    :return: exit code. 0 success, all others indicate failure
    """

    # Since we always have 3 dimensions when the corresponding caffe output
    # has less than 3 dimensions than those dimensions in Fathom are 1 and
    # have to be squeezed but we should not squeeze the myriad output when
    # caffe output has the last dimension = 1.
    while(len(result.shape) > len(expected.shape) and result.shape[-1] == 1):
        result = np.squeeze(result, len(result.shape) - 1)

    if validation_type == ValidationStatistic.accuracy_metrics:
        np.set_printoptions(precision=4, suppress=True)
        print("Result: ", result.shape)
        print(result)
        print("Expected: ", expected.shape)
        print(expected)
        if 0:   # Use this to compare the indicies for the max value observed (sample use case: Softmax output)
            print("Expected Max, @")
            print(np.max(expected), np.argmax(expected))
            print("Result Max,   @")
            print(np.max(result), np.argmax(result))
        result = np.reshape(result, expected.shape)

        compare_matricies(result, expected, filename)
        return 0
    elif validation_type == ValidationStatistic.top1:
        if (expected_index == None): #if exp id is not provided, get it from expected
            data = expected.flatten()
            ordered = np.argsort(data)
            ordered = ordered[::-1]
            expected_index = ordered[:1].flatten()
        exit_code = top_test(result, expected_index, 1)
        print("Result: ", result.shape)
        top_classifications(result, 1)
        print("Expected: ", expected.shape)
        top_classifications(expected, 1)
        compare_matricies(result, expected, filename)
        return exit_code
    elif validation_type == ValidationStatistic.top5:
        print("Result: ", result.shape)
        top_classifications(result)
        print("Expected: ", expected.shape)
        top_classifications(expected)
        compare_matricies(result, expected, filename)
        return 0
    elif validation_type == ValidationStatistic.class_check_exact:
        exit_code = significant_classification_check(
            result, expected, arguments.class_test_threshold, validation_type)
        return exit_code
    elif validation_type == ValidationStatistic.class_check_broad:
        exit_code = significant_classification_check(
            result, expected, arguments.class_test_threshold, validation_type)
        return exit_code
    elif validation_type == ValidationStatistic.ssd_pred_metric:
        if arguments.ma2480:
            result=np.squeeze(result,axis=0)
            expected=np.squeeze(expected,axis=0)
        exit_code = ssd_metrics(result, expected)
        return exit_code
    else:
        throw_error(ErrorTable.ValidationSelectionError)

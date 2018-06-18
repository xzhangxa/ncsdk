set(ILSVRC12_DIR ${CMAKE_CURRENT_LIST_DIR}/ilsvrc12)

download_extract(
    http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    ${ILSVRC12_DIR}
    "Download ilsvrc12"
    ilsvrc12
)
add_dependencies(examples ilsvrc12)

download(
    https://raw.githubusercontent.com/BVLC/caffe/master/python/caffe/imagenet/ilsvrc_2012_mean.npy
    ${ILSVRC12_DIR}
    "Download ilsvrc_2012_mean.npy"
    ilsvrc12_mean
)
add_dependencies(examples ilsvrc12_mean)

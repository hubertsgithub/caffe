import numpy as np
from caffe.classifier import Classifier

deploy_source = '...'
caffemodel_source = '...'

deploy_target = '...'
caffemodel_target = '...'

# Initialize target layers by copying from the source.  Note that for layers
# that are not initialized via a transfer will probably need to be manually
# initialized with the appropriate random values.
param_mapping = {
    'target layer...': 'source layer...',
    'target layer...': 'source layer...',
    'target layer...': 'source layer...',
    'target layer...': 'source layer...',
}

net_source = Classifier(deploy_source, caffemodel_source)
net_target = Classifier(deploy_target, caffemodel_source)

for t, s in param_mapping.iteritems():
    for blob_idx in (0, 1):
        print '%s %s %s <-- %s %s %s' % (
            t, blob_idx, net_target.params[t][blob_idx].data.shape,
            s, blob_idx, net_source.params[s][blob_idx].data.shape,
        )
        net_target.params[t][blob_idx].data[...] = (
            np.reshape(
                net_source.params[s][blob_idx].data,
                net_target.params[t][blob_idx].data.shape
            )
        )

net_target.save(caffemodel_target)

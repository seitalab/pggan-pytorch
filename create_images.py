import os
import torch
from config import config
import torchvision.utils as vutils


use_cuda = True
checkpoint_path = 'repo/model/gen_R8_T3400.pth.tar'
num_images = 1000  # should be multiple of 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load trained model.
import network as net
test_model = net.Generator(config)
if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.device_count() > 1:
    test_model = torch.nn.DataParallel(test_model)
test_model.to(device)

for resl in range(3, 8+1):
    test_model.module.grow_network(resl)
    test_model.module.flush_network()
print(test_model)


print('load checkpoint form ... {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path)
test_model.module.load_state_dict(checkpoint['state_dict'])
test_model.to(device)

save_path = 'repo/save/test/chest256'
if not os.path.exists(save_path):
    os.system('mkdir -p {}'.format(save_path))

test_model.eval()
with torch.no_grad():
    for i in range(100):
        zs = torch.FloatTensor(num_images // 100, config.nz).normal_(0.0, 1.0).to(device)
        images = test_model.module(zs)
        for j, data in enumerate(images.detach()):
            img_path = os.path.join(save_path, '%04d.png' % (i * num_images // 100 + j))
            vutils.save_image(data, img_path, normalize=True)
'''
# create folder.
for i in range(1000):
    name = 'repo/interpolation/try_{}'.format(i)
    if not os.path.exists(name):
        os.system('mkdir -p {}'.format(name))
        break;

# interpolate between twe noise(z1, z2).
z_intp = torch.FloatTensor(1, config.nz)
z1 = torch.FloatTensor(1, config.nz).normal_(0.0, 1.0)
z2 = torch.FloatTensor(1, config.nz).normal_(0.0, 1.0)
if use_cuda:
    z_intp = z_intp.cuda()
    z1 = z1.cuda()
    z2 = z2.cuda()
    test_model = test_model.cuda()

z_intp = Variable(z_intp)


for i in range(1, n_intp+1):
    alpha = 1.0/float(n_intp+1)
    z_intp.data = z1.mul_(alpha) + z2.mul_(1.0-alpha)
    fake_im = test_model.module(z_intp)
    fname = os.path.join(name, '_intp{}.jpg'.format(i))
    utils.save_image_single(fake_im.data, fname, imsize=pow(2,config.max_resl))
    print('saved {}-th interpolated image ...'.format(i))



self.z1.data.normal_(0.0, 1.0)
self.z2 = torch.FloatTensor(1, config.nz).cuda() if use_cuda else torch.FloatTensor(1,config.nz)
self.z2 = Variable(self.z2)
self.z2.data.normal_(0.0, 1.0)

print
'''

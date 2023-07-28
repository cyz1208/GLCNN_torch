#coding=gbk

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import os
import pandas as pd
from torchvision.utils import save_image
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ���û���ļ��оʹ���һ���ļ���
sample_dir = 'C:/Users/chenyuzhuo/Desktop/ML/scripts/torch_test/vae_samples'
if not os.path.exists(sample_dir):
	os.makedirs(sample_dir)
else:
	[os.remove(os.path.join(sample_dir, filename)) for filename in os.listdir(sample_dir)
	 if "sampled" in filename or "reconst" in filename]

# Hyper-parameters
h1_dim = 6   # ���������
h2_dim = 16
h3_dim = 120
h4_dim = 2000  # ȫ���Ӳ㳤��
h5_dim = 200
h6_dim = 200
z_dim = 10
num_epochs = 50
batch_size = 128 * 2
learning_rate = 1e-3

num_channels = 6   # ѵ���õ�ͨ����Ŀ
kl_weight = 0.5   # KLɢ�ȵ�Ȩ��

# ɸѡѵ����
# df = pd.read_csv("C:/Users/chenyuzhuo/Desktop/ML/scripts/torch_test/data_2/id_prop.csv")
# df_names = pd.read_csv("C:/Users/chenyuzhuo/Desktop/ML/scripts/torch_test/E_OH_all.csv")
# selected_ids = []
# for _i, d in df_names.iterrows():
# 	if "HV" not in d["sub"]:   # mesh,add_N,sub,metal
# 		selected_ids.append(_i)
#
# df_new = pd.DataFrame(columns=df.columns)
# for _i, d in df.iterrows():
# 	if int(d["id"].split("-")[0]) in selected_ids:
# 		df_new = df_new.append(d, ignore_index=True)
# df_new.to_csv("C:/Users/chenyuzhuo/Desktop/ML/scripts/torch_test/data_2/id_prop_SV_DV.csv", index=False)
# exit()


class CustomImageDataset(Dataset):
	def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
		self.img_labels = pd.read_csv(annotations_file)
		self.img_dir = img_dir
		self.transform = transform
		self.target_transform = target_transform

	def __len__(self):
		return len(self.img_labels)

	def __getitem__(self, idx):
		img_path = self.img_dir + '/' + self.img_labels.iloc[idx, 0] + '.pkl'
		res = open(img_path, 'rb')
		# image = pickle.load(res)[1]  # ֻ��ȡĳһͨ��
		image = pickle.load(res)  # ��ȡȫ��ͨ��
		label = self.img_labels.iloc[idx, 1]
		if self.transform:
			image = self.transform(image)
		if self.target_transform:
			label = self.target_transform(label)
		return image, label


dataset = CustomImageDataset(annotations_file="C:/Users/chenyuzhuo/Desktop/ML/scripts/torch_test/data_2/id_prop_SV_DV.csv",
                             img_dir='C:/Users/chenyuzhuo/Desktop/ML/scripts/torch_test/data_2/')
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


class Reshape(nn.Module):
	def __init__(self, *args):
		super(Reshape, self).__init__()
		self.shape = args

	def forward(self, x):
		# ������ݼ����һ��batch��������С�ڶ����batch_batch��С�������mismatch���⡣
		# �����Լ��޸��£���ֻ��������shape��Ȼ��ͨ��x.szie(0)�������롣
		return x.view(self.shape)


# VAE model(CNN)
class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		self.conv1 = nn.Conv2d(num_channels, h1_dim, kernel_size=5, stride=1, padding=0)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
		self.bn1 = nn.BatchNorm2d(h1_dim)

		self.conv2 = nn.Conv2d(h1_dim, h2_dim, kernel_size=5, stride=1, padding=0)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
		self.bn2 = nn.BatchNorm2d(h2_dim)

		self.conv3 = nn.Sequential(
			nn.Conv2d(h2_dim, h3_dim, kernel_size=5, stride=1, padding=0),  # [128,120,1,1]
			nn.Flatten(),  # ��ֵ[128,120]
			nn.Linear(h3_dim, h4_dim),
			nn.ReLU(),
			nn.Dropout(),
			# nn.Linear(h4_dim, h5_dim),
			# nn.ReLU(),
			# nn.Dropout(),
			nn.Linear(h4_dim, z_dim),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(h2_dim, h3_dim, kernel_size=5, stride=1, padding=0),  # [128,120,1,1]
			nn.Flatten(),  # ����
			nn.Linear(h3_dim, h4_dim),
			nn.ReLU(),
			nn.Dropout(),
			# nn.Linear(h4_dim, h5_dim),
			# nn.ReLU(),
			# nn.Dropout(),
			nn.Linear(h4_dim, z_dim),
		)

		self.FCTrans = nn.Sequential(
			nn.Linear(z_dim, h4_dim),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(h4_dim, h3_dim),
			# nn.ReLU(),
			# nn.Dropout(),
			# nn.Linear(h4_dim, h3_dim),
			Reshape(-1, h3_dim, 1, 1),  # [128,120,1,1]
		)

		self.convtrans1 = nn.ConvTranspose2d(h3_dim, h2_dim, kernel_size=5, stride=1, padding=0)  # [128,16,5,5]
		self.pooltrans1 = nn.MaxUnpool2d(kernel_size=2, stride=2)   # [128,16,10,10]

		self.convtrans2 = nn.ConvTranspose2d(h2_dim, h1_dim, kernel_size=5, stride=1, padding=0)  # [128,6,14,14]
		self.pooltrans2 = nn.MaxUnpool2d(kernel_size=2, stride=2)   # [128,6,28,28]

		self.convtrans3 = nn.ConvTranspose2d(h1_dim, num_channels, kernel_size=5, stride=1, padding=0)  # [128,6,32,32]

	# �������
	def encode(self, x):
		h1 = self.conv1(x)  # [128,6,28,28]
		h1, indices_1 = self.pool1(h1)  # [128,6,14,14]
		h1 = self.bn1(h1)
		h1 = F.relu(h1)

		h2 = self.conv2(h1)  # [128,6,28,28]
		h2, indices_2 = self.pool2(h2)  # [128,6,14,14]
		h2 = self.bn2(h2)
		h2 = F.relu(h2)

		return self.conv3(h2), self.conv4(h2), indices_1, indices_2

	# �ز�����
	def reparameterize(self, mu, log_var):
		std = torch.exp(log_var / 2)
		eps = torch.randn_like(std)
		return mu + eps * std

	# �������
	def decode(self, z, indices_1, indices_2):
		h1 = self.FCTrans(z)
		h1 = self.convtrans1(h1)
		h1 = self.pooltrans1(h1, indices_2)
		h1 = F.relu(h1)

		h2 = self.convtrans2(h1)
		h2 = self.pooltrans2(h2, indices_1)
		h2 = F.relu(h2)
		output = self.convtrans3(h2)

		return output

	# ǰ�򴫲�
	def forward(self, x):
		mu, log_var, indices_1, indices_2 = self.encode(x)
		z = self.reparameterize(mu, log_var)
		x_reconst = self.decode(z, indices_1, indices_2)
		return x_reconst, mu, log_var, indices_1, indices_2


# ʵ����
model = VAE().to(device)

# �����Ż���
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

import time

for epoch in range(num_epochs):
	start = time.perf_counter()   	# �����ʱ
	indices_1, indices_2 = None, None
	for i, (x, y) in enumerate(data_loader):
		# ��ȡ��������ǰ�򴫲�
		x = x.to(device).view(-1, num_channels, 32, 32)
		x_reconst, mu, log_var, tmp_1, tmp_2 = model(x)
		if i == len(data_loader) - 2:   # indicesѡȡ�����ڶ���batch������
			indices_1, indices_2 = tmp_1, tmp_2

		# �����ع���ʧ��KLɢ�ȣ�KLɢ�����ں������ֲַ������Ƴ̶ȣ�
		# reconst_loss = F.binary_cross_entropy(x_reconst, x.view(-1, num_channels, 32, 32), size_average=False)
		reconst_loss = F.mse_loss(x_reconst, x.view(-1, num_channels, 32, 32), size_average=False)
		kl_div = - kl_weight * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

		# ���򴫲����Ż�
		loss = reconst_loss + kl_div
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % 100 == 0 or (i + 1) == len(data_loader):
			print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}."
			      .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()))

	# ����ѵ����ģ�ͽ��в���
	with torch.no_grad():
		# ������ɵ�ͼ��
		z = torch.randn(batch_size, z_dim).to(device)
		out = model.decode(z, indices_1, indices_2).view(-1, num_channels, 32, 32)
		out = out[:, [1], :, :]   # ֻ��ȡĳһͨ��
		save_image(out, os.path.join(sample_dir, 'sampled_{}.png'.format(epoch + 1)))

		# �ع���ͼ��
		out, _, _, _, _ = model(x)
		x, out = x[:, [1], :, :], out[:, [1], :, :]   # ֻ��ȡĳһͨ��
		x_concat = torch.cat([x.view(-1, 1, 32, 32), out.view(-1, 1, 32, 32)], dim=3)
		save_image(x_concat, os.path.join(sample_dir, 'reconst_{}.png'.format(epoch + 1)))

	end = time.perf_counter()
	print("time consumed: {:.4f}".format(end - start))


work_dir: ./work_dir/ctrgcn_joint_tta
# weights: ./work_dir/uav/ctrgcn_joint/runs-65-8450.pt
weights: ./work_dir/ctrgcn_joint/runs-38-9728.pt
# feeder
feeder: feeders.feeder_uav_tta.Feeder
train_feeder_args:
  data_path: /root/MS-CTR-GCN/data/train_joint.npy
  label_path: /root/MS-CTR-GCN/data/train_label.npy
  debug: False
  random_choose: True
  random_shift: True
  random_move: True
  window_size: 120
  normalization: True
  random_rot: False
  p_interval: [0.95]
  vel: False
  bone: False

test_feeder_args:
  data_path: /root/MS-CTR-GCN/data/test_joint_A.npy
  label_path: /root/MS-CTR-GCN/data/test_label_A.npy
  window_size: 120
  p_interval: [0.95]
  vel: False
  bone: False
  debug: False
  normalization: True

# model
model: model.ctrgcn.Model
model_args:
  num_class: 155
  num_point: 17
  num_person: 2
  graph: graph.uav_human.Graph
  graph_args:
    labeling_mode: 'spatial'

#optim
weight_decay: 0
base_lr: 1e-4
min_lr: 1e-6
lr_decay_rate: 0.1
step: [35, 55]
warm_up_epoch: 0

# training
device: [0]
batch_size: 64
test_batch_size: 64
num_epoch: 80
nesterov: False
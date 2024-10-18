import numpy as np
import pickle

# # 处理B数据的虚拟label格式
# a = np.load('./data/test_A_label.npy')
# print(a.shape)
# b = np.load('./data/test_joint_B.npy')
# a = np.tile(a[:1], b.shape[0])
# print(a.shape)
# np.save('./data/test_B_label.npy', a)

with open('./work_dir/2996/epoch1_test_score.pkl', 'rb') as f:
    data = pickle.load(f)
    print('生成提交结果文件到目录: ./work_dir/2996/pred.npy')
    print(data)
    print(data.shape)

    # 保存为 npy 文件
    np.save('./work_dir/2996/pred.npy', data)
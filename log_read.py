import os
import matplotlib.pyplot as plt

log_path = 'model_checkpoint/HanBart_202110220849/info/'
log_files = os.listdir(log_path)

log_file_name = ['train_instance_'+str(i)+'_info.log'  for i in sorted([int(log.split('_')[2]) for log in log_files]) ]

train_loss = []
for log_file in log_file_name:
    with open(log_path+log_file, 'r') as f:
        log_data = f.readlines()
        
    for log in log_data:
        if log.split('|')[-1].startswith(' Loss'):
            train_loss.append(float(log.split('|')[-1].split(' ')[2]))

title = log_path.split('/')[1]
plt.plot(train_loss)
plt.title(title)

plt.savefig('./loss_graph.png')

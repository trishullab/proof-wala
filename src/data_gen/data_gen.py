#from itp_interface.main.run_tool import main
from itp_interface.tools.training_data import TrainingData

if __name__ == '__main__':
    td = TrainingData(folder='/home/amthakur/Projects/thrall/.log/train', training_meta_filename='local.meta.json')
    td.load()
    data = [x for x in td]
    print(len(data))
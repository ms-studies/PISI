from data_loader import loadTaskData
import numpy as np

def main():
    taskData = loadTaskData()
    taskData['review'] = np.random.randint(0, 6, taskData.shape[0])
    # print(taskData)
    taskData.to_csv('submission.csv', sep=';', index=False, header=False)

if __name__ == '__main__':
    main()
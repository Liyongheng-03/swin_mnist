import matplotlib.pyplot as plt

# 训练和验证的损失和准确率数据
epochs = range(10)
train_loss = [0.222, 0.111, 0.086, 0.070, 0.061, 0.052, 0.046, 0.044, 0.040, 0.036]
valid_loss = [0.134, 0.095, 0.083, 0.093, 0.064, 0.058, 0.062, 0.062, 0.053, 0.056]
train_acc = [0.930, 0.965, 0.973, 0.978, 0.981, 0.984, 0.985, 0.986, 0.988, 0.988]
valid_acc = [0.958, 0.970, 0.974, 0.971, 0.978, 0.983, 0.981, 0.982, 0.984, 0.984]

# 绘制训练和验证损失曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, valid_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制训练和验证准确率曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, valid_acc, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
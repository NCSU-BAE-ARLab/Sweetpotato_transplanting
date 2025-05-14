import matplotlib.pyplot as plt
import numpy as np

class MetricsPlotter:
    def __init__(self,result_dir='../'):
        self.epochs = []
        self.losses = []
        self.ap_values = []  # Average Precision
        self.ar_values = []  # Average Recall
        self.result_dir = result_dir
        plt.ion()  # Enable interactive mode

    def update(self, epoch, loss, ap, ar):
        """ Update metrics and refresh plots. """
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.ap_values.append(ap)
        self.ar_values.append(ar)
        self._plot()

    def _plot(self):
        """ Create and update the plots. """
        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.epochs, self.losses, marker='o', linestyle='-', label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss over Epochs")
        plt.legend()
        plt.grid()

        # AP/AR plot
        plt.subplot(1, 2, 2)
        plt.plot(self.epochs, self.ap_values, marker='o', linestyle='-', label="Average Precision (AP)")
        plt.plot(self.epochs, self.ar_values, marker='s', linestyle='-', label="Average Recall (AR)")
        plt.xlabel("Epoch")
        plt.ylabel("Metric Value")
        plt.title("COCO Evaluation Metrics over Epochs")
        plt.legend()
        plt.grid()

        plt.pause(0.1)  # Update plots
        plt.savefig(self.result_dir+"/training_metrics.png")  # Save the figure

    def show_final(self):
        """ Keep the plots visible after training ends. """
        plt.ioff()
        plt.show()

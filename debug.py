import src.lightly_train as lightly_train

def main():
    lightly_train.train(
        out="out/my_experiment1",  # Output directory
        data="/Users/jonaswurst/Lightly/dataset_clothing_images/images",  # Directory with images
        model="torchvision/resnet18",  # Model to train
        epochs=3,  # Number of epochs to train
        batch_size=32,  # Batch size
        accelerator="cpu",
        callbacks={"model_export": {"every_n_epochs": 2}}
    )

if __name__ == "__main__":
    main()
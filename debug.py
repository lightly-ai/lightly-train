import src.lightly_train as lightly_train

def main():
    lightly_train.train(
        out="out/my_experiment2",  # Output directory
        data="/Users/jonaswurst/Lightly/dataset_clothing_images/images",  # Directory with images
        model="dinov2_vit/vitl14",  # Model to train
        epochs=3,  # Number of epochs to train
        batch_size=32,  # Batch size
        accelerator="cpu",
        model_args={
            "pretrained": True
        }, 
    )

if __name__ == "__main__":
    main()
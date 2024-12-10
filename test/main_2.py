from mh_training import train_with_mh
import torch
import sys

if __name__ == "__main__":
    try:
        # Set random seeds
        torch.manual_seed(42)
        
        # Run training with simplified parameters
        trained_policy = train_with_mh(episodes=1000)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        sys.exit(0)  # Clean exit
from utils import get_cifar10_loaders
try:
    print("Getting loaders...")
    loaders = get_cifar10_loaders(3, 4)
    print("Loaders got.")
    train, test, global_train = loaders
    print("Iterating...")
    for x, y in train[0]:
        print(x.size())
        break
    print("Done.")
except Exception as e:
    print(f"Error: {e}")

from neural_network.main import train, test

loss = train(2,2, 0.001)

test = test()
print(test)

print (f"loss: {loss}")
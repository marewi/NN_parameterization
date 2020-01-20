from neural_network.main import train

try:
    test_reward = train(1, 15, 0.428)
except Exception as e:
    print(str(e))

# if test_reward == 'nan':
#     print("Error caught")

print(test_reward)


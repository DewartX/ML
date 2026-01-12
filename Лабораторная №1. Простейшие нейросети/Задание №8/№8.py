def neuralNetwork(inp, weights):

    prediction = [0,0]

    for i in range(len(weights)):

        ws=0 

        for j in range(len(inp)):

            ws += inp[j] * weights[i][j]

        prediction[i] = ws

    return prediction


inp = [50, 165]


weights_1 = [0.2, 0.1]

weights_2 = [0.2, 0.1]

weights = [weights_1, weights_2]

print(neuralNetwork(inp,weights))
using HDF5
using LinearAlgebra

test = h5open("test.h5","r")
train = h5open("train.h5","r")
test_set_x_orig = read(test["test_set_x"])
test_set_y = read(test["test_set_y"])
train_set_x_orig = read(train["train_set_x"])
train_set_y = read(train["train_set_y"])
classes = read(test["list_classes"])
m_train = size(train_set_x_orig,4)
m_test = size(test_set_x_orig,4)

train_set_x_flatten = reshape(train_set_x_orig,:,m_train)
test_set_x_flatten = reshape(test_set_x_orig,:,m_test)
train_set_y = reshape(train_set_y, (1,size(train_set_y)...))
test_set_y = reshape(test_set_y, (1,size(test_set_y)...))

println(string("train_set_x_flatten shape: ", size(train_set_x_flatten)))
println(string("train_set_y shape: ", size(train_set_y)))
println(string("test_set_x_flatten shape: ",size(test_set_x_flatten)))
println(string("test_set_y shape: ", size(test_set_y)))

train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

function sigmoid(z::Array)
    _sigmoid(z::Real) = 1.0 / (1.0 + exp(-z))
    return broadcast(_sigmoid,z)
end

function propagate(w, b, X, Y)
    m = size(X,2)
    A = sigmoid( w' * X  .+ b)
    dw = (X * (A-Y)')/m
    db = sum(A-Y)/m
    cost = -sum((Y.*log.(A)) + (1 .-Y) .* log.(1 .- A))/m
    return cost, dw, db
end

function optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=false)
    db = 0.0
    dw = zeros(size(w,1),1)
    cost = 0.0
    for i in 1:num_iterations
        cost, dw, db = propagate(w,b,X,Y)
        w = w .- (learning_rate .* dw)
        b = b - learning_rate * db
        if print_cost && mod(i,100) == 0
            println(string("Cost after iteration ", i, " ", cost))
        end
    end
    return cost, dw, db, w, b
end

function predict(w, b, X)
    m = size(X,2)
    Y_prediction = zeros(1,m)
    
    A = sigmoid((w' * X) .+ b)
    for i in 1:size(A,2)
        if A[1,i] > 0.5
            Y_prediction[1,i] = 1
        else
            Y_prediction[1,i] = 0
        end
    end
    return Y_prediction
end

function model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=false)
    w = zeros(size(X_train,1),1)
    b = 0.0

    cost, dw, db, w, b = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    println("train accuracy: ", 100-(sum(abs.(Y_prediction_train .- Y_train))/size(Y_train,2))*100)
    println("test accuracy: ", 100-(sum(abs.(Y_prediction_test .- Y_test))/size(Y_test,2))*100)    
 end

model(train_set_x, train_set_y, test_set_x, test_set_y, 2000, 0.005, true)

import numpy as np
from numpy import pi as PI
from matplotlib import pyplot as plt
from scipy.stats import unitary_group
from scipy.linalg import norm

import paddle
from paddle_quantum.ansatz import Circuit
from paddle_quantum.linalg import dagger

# Image processing package PIL
from PIL import Image



# Open the picture prepared in advance
img = Image.open('../figures/MNIST_32.png')
imgmat = np.array(list(img.getdata(band=0)), float)
imgmat.shape = (img.size[1], img.size[0])
imgmat = np.matrix(imgmat)/255


# Convert the image into numpy array
def Mat_generator():
    imgmat = np.array(list(img.getdata(band=0)), float)
    imgmat.shape = (img.size[1], img.size[0])
    lenna = np.matrix(imgmat)
    return lenna.astype('complex128')


# Set circuit parameters
cir_depth = 40      # depth of circuit
num_qubits = 5      # Number of qubits

# Hyper-parameters
RANK = 8            # Set the number of rank you want to learn
ITR = 10           # Number of iterations
LR = 0.02           # Learning rate
SEED = 14           # Random number seed

# Set the learning weight
weight = np.arange(2 * RANK, 0, -2).astype('complex128')

M_err = Mat_generator()



class VQSVD():
    def __init__(self, matrix: np.ndarray, weights: np.ndarray, num_qubits: int, depth: int, rank: int, lr: float, itr: int, seed: int):
        
        # Hyperparameters
        self.rank = rank
        self.lr = lr
        self.itr = itr
        
        paddle.seed(seed)
        
        # Create the parameter theta for learning U
        self.cir_U = self.U_theta(num_qubits, depth)
        
        # Create a parameter phi to learn V_dagger
        self.cir_V = self.U_theta(num_qubits, depth)
        
        # Convert Numpy array to Tensor supported in Paddle
        self.M = paddle.to_tensor(matrix)
        self.weight = paddle.to_tensor(weights)
        
    # Define circuit of quantum neural network
    def U_theta(self,num_qubits: int, depth: int) -> Circuit:

        # Initialize the network with Circuit
        cir = Circuit(num_qubits)
        
        # Build a hierarchy：
        for _ in range(depth):
            cir.ry()
            cir.cnot()

        return cir

    # Define the loss function
    def loss_func(self):
        
        # Get the unitary matrix representation of the quantum neural network
        U = self.cir_U.unitary_matrix()
        V = self.cir_V.unitary_matrix()
    
        # Initialize the loss function and singular value memory
        loss = paddle.to_tensor(0.0)
        singular_values = np.zeros(self.rank)
        
        # Define loss function
        for i in range(self.rank):
            loss -= paddle.real(self.weight)[i] * paddle.real(dagger(U) @ self.M @ V)[i][i]
            singular_values[i] = paddle.real(dagger(U) @ self.M @ V)[i][i].numpy()
        
        # Function returns learned singular values and loss function
        return loss, singular_values
    
    def get_matrix_U(self):
        return self.cir_U.unitary_matrix()
    
    def get_matrix_V(self):
        return self.cir_V.unitary_matrix()
    
    # Train the VQSVD network
    def train(self):
        loss_list, singular_value_list = [], []
        optimizer = paddle.optimizer.Adam(learning_rate=self.lr, parameters=self.cir_U.parameters()+self.cir_V.parameters())
        for itr in range(self.itr):
            loss, singular_values = self.loss_func()
            loss.backward()
            optimizer.minimize(loss)
            optimizer.clear_grad()
            loss_list.append(loss.numpy()[0])
            singular_value_list.append(singular_values)
            if itr% 10 == 0:
                print('iter:', itr,'loss:','%.4f'% loss.numpy()[0])
                
        return loss_list, singular_value_list
    

# Record the optimization process
loss_list, singular_value_list = [], []
U_learned, V_dagger_learned = [], []
    
# Construct the VQSVD network and train
net = VQSVD(matrix=Mat_generator(), weights=weight, num_qubits=num_qubits, depth=cir_depth, rank=RANK, lr=LR, itr=ITR, seed=SEED)
loss_list, singular_value_list = net.train()

# Record the last two unitary matrices learned
U_learned = net.get_matrix_U().numpy()
V_dagger_learned = dagger(net.get_matrix_V()).numpy()

singular_value = singular_value_list[-1]
mat = np.matrix(U_learned.real[:, :RANK]) * np.diag(singular_value[:RANK])* np.matrix(V_dagger_learned.real[:RANK, :])

reconstimg = mat
plt.imshow(reconstimg, cmap='gray')
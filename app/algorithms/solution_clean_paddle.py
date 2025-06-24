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

    def frobenius_norm_error(self, M, M_err, U_learned, singular_value_list, V_dagger_learned, RANK):
        err_local, err_subfull, err_SVD = [], [], []
        U, D, V_dagger = np.linalg.svd(M, full_matrices=True)
        # Use the last learned singular values
        singular_value = np.array(singular_value_list[-1])
        for i in range(1, RANK + 1):
            lowrank_mat = U[:, :i] @ np.diag(D[:i]) @ V_dagger[:i, :]
            recons_mat = U_learned[:, :i] @ np.diag(singular_value[:i]) @ V_dagger_learned[:i, :]
            err_local.append(norm(lowrank_mat - recons_mat))
            err_subfull.append(norm(M_err - recons_mat))
            err_SVD.append(norm(M_err - lowrank_mat))
        return err_local, err_subfull, err_SVD


class LossPlot:
    def __init__(self):
        self.route = '../results/'

    def loss_plot(self, loss):
        '''
        Loss is a list, this function plots loss over iteration
        '''
        plt.plot(list(range(1, len(loss)+1)), loss)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('VQSVD Loss Learning Curve')
        plt.grid(True)        
        self.save_plot('PD_figure_01.png')
        plt.show()
        plt.close()
        
    
    def plot_singular_values_comparison(self, params:dict):
        RANK = params.get('rank')
        err_subfull = params.get('err_subfull')
        err_SVD = params.get('err_SVD')        
        fig, ax = plt.subplots()
        ax.plot(list(range(1, RANK+1)), err_subfull, "o-.", 
                label = 'Reconstruction via VQSVD')
        ax.plot(list(range(1, RANK+1)), err_SVD, "^--", 
                label='Reconstruction via SVD')
        plt.xlabel('Singular Value Used (Rank)', fontsize = 14)
        plt.ylabel('Norm Distance', fontsize = 14)
        leg = plt.legend(frameon=True)
        leg.get_frame().set_edgecolor('k')
        self.save_plot('PD_figure_02.png')
        plt.show()
        plt.close()
     
    def plot_matrix_as_image(self, M, title="Reconstruction"):
        plt.imshow(M, cmap="gray")
        plt.title(title)
        plt.colorbar()
        self.save_plot('PD_figure_03.png')
        plt.show()
        plt.close()

    def save_plot(self, filename):
        plt.savefig(self.route + filename)
        print(f"Plot saved as {self.route + filename}")


class MatGenerator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

    def from_image(self, image_path):
        """
        Genera una matriz a partir de una imagen en escala de grises.
        """
        img = Image.open(image_path)
        imgmat = np.array(list(img.getdata(band=0)), float)
        imgmat.shape = (img.size[1], img.size[0])
        imgmat = np.matrix(imgmat) / 255
        return imgmat.astype('complex128')

    def random_matrix(self):
        """
        Genera una matriz cuadrada aleatoria compleja de tamaño 2**num_qubits.
        """
        size = 2 ** self.num_qubits
        real_part = np.random.randint(10, size=(size, size))
        imag_part = np.random.randint(10, size=(size, size))
        M = real_part + 1j * imag_part
        return M.astype('complex128')




# Set circuit parameters
cir_depth = 40      # depth of circuit
num_qubits = 5      # Number of qubits

# Hyper-parameters
RANK = 8            # Set the number of rank you want to learn
ITR = 100          # Number of iterations
LR = 0.02           # Learning rate
SEED = 14           # Random number seed

# Set the learning weight
weight = np.arange(2 * RANK, 0, -2).astype('complex128')


loss_list, singular_value_list = [], []
U_learned, V_dagger_learned = [], []
matrix_gen = MatGenerator(num_qubits)
mat = matrix_gen.from_image('../figures/MNIST_32.png')


net = VQSVD(matrix=mat, weights=weight, num_qubits=num_qubits, depth=cir_depth, rank=RANK, lr=LR, itr=ITR, seed=SEED)
loss_list, singular_value_list = net.train()


# Record the last two unitary matrices learned
U_learned = net.get_matrix_U().numpy()
V_dagger_learned = net.get_matrix_V().numpy().conj().T


plot = LossPlot()
plot.loss_plot(loss_list)

#def random_M_generator(num_qubits):
#    return np.random.randint(10, size = (2**num_qubits, 2**num_qubits)) + 1j*np.random.randint(10, size = (2**num_qubits, 2**num_qubits))

M = matrix_gen.random_matrix()
M_err = np.copy(M)
err_local, err_subfull, err_SVD = net.frobenius_norm_error(net.M, M_err, U_learned, singular_value_list, V_dagger_learned, RANK)
plot.plot_singular_values_comparison(params={'rank': RANK, 'err_subfull': err_subfull, 'err_SVD': err_SVD})

singular_value = singular_value_list[-1]
mat = np.matrix(U_learned.real[:, :RANK]) * np.diag(singular_value[:RANK])* np.matrix(V_dagger_learned.real[:RANK, :])
plot.plot_matrix_as_image(mat, title="Reconstruction via VQSVD")



















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




M_err = Mat_generator()






# Record the last two unitary matrices learned
U_learned = net.get_matrix_U().numpy()
V_dagger_learned = dagger(net.get_matrix_V()).numpy()

singular_value = singular_value_list[-1]
mat = np.matrix(U_learned.real[:, :RANK]) * np.diag(singular_value[:RANK])* np.matrix(V_dagger_learned.real[:RANK, :])

reconstimg = mat
plt.imshow(reconstimg, cmap='gray')
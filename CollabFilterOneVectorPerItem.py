'''
CollabFilterOneVectorPerItem.py

Defines class: `CollabFilterOneVectorPerItem`

Scroll down to __main__ to see a usage example.
'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets
import matplotlib.pyplot as plt

# Some packages you might need (uncomment as necessary)
## import pandas as pd
## import matplotlib

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterOneVectorPerItem(AbstractBaseCollabFilterSGD):
    ''' One-vector-per-user, one-vector-per-item recommendation model.

    Assumes each user, each item has learned vector of size `n_factors`.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)
    b_per_user : 1D array, size n_users
    c_per_item : 1D array, size n_items
    U : 2D array, size n_users x n_factors
    V : 2D array, size n_items x n_factors

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values
        '''
        random_state = self.random_state # inherited RandomState object

        # TODO fix the lines below to have right dimensionality & values
        # TIP: use self.n_factors to access number of hidden dimensions
        self.param_dict = dict(
            mu=ag_np.array(random_state.randn(1)),
            b_per_user=ag_np.array(random_state.randn(n_users)),
            c_per_item=ag_np.array(random_state.randn(n_items)),
            U=random_state.randn(n_users, self.n_factors),
            V=random_state.randn(n_items, self.n_factors),
        )


    def predict(self, user_id_N, item_id_N,
                mu=None, b_per_user=None, c_per_item=None, U=None, V=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic

        # handles if parameters are not provided, falling back to initialized values
        mu = mu if mu is not None else self.param_dict['mu']
        b_per_user = b_per_user if b_per_user is not None else self.param_dict['b_per_user']
        c_per_item = c_per_item if c_per_item is not None else self.param_dict['c_per_item']
        U = U if U is not None else self.param_dict['U']
        V = V if V is not None else self.param_dict['V']

        u_i = U[user_id_N]
        v_j = V[item_id_N]
        uv_sum = ag_np.sum(u_i * v_j, axis=1)

        b_i = b_per_user[user_id_N]
        c_j = c_per_item[item_id_N]
        yhat_N = mu + b_i + c_j + uv_sum
        
        return yhat_N


    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        # TIP: use self.alpha to access regularization strength
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)

        sumVsquared = ag_np.sum(ag_np.square(param_dict['V']))
        sumUsquared = ag_np.sum(ag_np.square(param_dict['U']))
        alphaTerm = self.alpha * (sumVsquared + sumUsquared)

        errorTerm = ag_np.sum(ag_np.square(y_N - yhat_N))

        loss_total = alphaTerm + errorTerm
        return loss_total    


if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model1 = CollabFilterOneVectorPerItem(
        n_epochs=100, batch_size=1000, step_size=0.5,
        n_factors=50, alpha=0.0, random_state=42)
    model1.init_parameter_dict(n_users, n_items, train_tuple)
    model1.fit(train_tuple, valid_tuple)
    plt.plot(model1.trace_epoch, model1.trace_rmse_valid, label='Val RMSE')
    plt.plot(model1.trace_epoch, model1.trace_rmse_train, label='Train RMSE')
    plt.legend()
    plt.show()

    


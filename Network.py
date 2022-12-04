import numpy as np
import pandas as pd
import scipy
from scipy import stats

class Network:
    """
    A simulated supply chain network.
    
    Inputs:
        n : int - number of entities
        p_c : float - proportion of companies        
        p_3 : float - proportion of third-party suppliers
        p_4 : float - proportion of fourth-party suppliers
        d_size_c : tuple - distribution of company sizes
        d_size_3 : tuple - distribution of third-party sizes
        d_size_4 : tuple - distribution of fourth-party sizes
        d_conn_c : tuple - distribution of company outgoing connections
        d_conn_3 : tuple - distribution of third-party outgoing connections
        random_state : int - numpy random seed
        
    Outputs:
        self.network : dataframe - supply chain network
        
    User must specify the value for n, number of entities.
    The distribution variables are tuples of the form (location, scale, upper bound).
    Each distribution follows the scipy half-normal continuous distribution.
    """
    
    def __init__(self,
                 n,
                 p_c = .98,
                 p_3 = .015,
                 p_4 = .005,
                 d_size_c = (1,20000,1000000),
                 d_size_3 = (1,50000,1000000),
                 d_size_4 = (50,1000,10000),
                 d_conn_c = (1,30,1000),
                 d_conn_3 = (1,90,1000),
                 random_state = None):
        """
        Initialize network.
        """
        # Network Parameters
        self.n = n
        self.p_c = p_c
        self.p_3 = p_3
        self.p_4 = p_4
        self.d_size_c = d_size_c
        self.d_size_3 = d_size_3
        self.d_size_4 = d_size_4
        self.d_conn_c = d_conn_c
        self.d_conn_3 = d_conn_3
        assert self.p_c + self.p_3 + self.p_4 == 1, \
               "Proportion of companies, third-parties, and fourth-parties must equal 1."
        # Random State
        if random_state != None:
            np.random.seed(random_state)
        # Class Methods
        self.generate_IDs()
        self.generate_sizes()
        self.generate_outgoing_connections()
        self.generate_incoming_connections()
        self.create_dataframe()
        
    def generate_IDs(self):
        """
        Generate unique IDs for every entity in the network.
        """
        # Calculate number of companies, third-parties, and fourth-parties
        number_company = int(np.ceil(self.p_c*self.n))
        number_3rdParty = int(np.ceil(self.p_3*self.n))
        number_4thParty = int(np.ceil(self.p_4*self.n))
        # Ensure number of entities equals self.n
        while number_company + number_3rdParty + number_4thParty > self.n:
            number_company -= 1
        # Generate unique IDs for every entity
        self.company_IDs = list(range(number_company))
        self._3rdParty_IDs = list(range(number_company,number_company+number_3rdParty))
        self._4thParty_IDs = list(range(number_company+number_3rdParty,
                                        number_company+number_3rdParty+number_4thParty))
        
    def generate_sizes(self):
        """
        Generate sizes for every entity in the network.
        """
        # Generate sizes for every company
        company_sizes = []
        for _ in self.company_IDs:
            x = np.linspace(0,self.d_size_c[2],10000)
            size = np.random.choice(x,
                                    size = 1,
                                    p = scipy.stats.halfnorm.pdf(x,self.d_size_c[0],self.d_size_c[1])/
                                        np.sum(scipy.stats.halfnorm.pdf(x,self.d_size_c[0],self.d_size_c[1])))[0]
            company_sizes.append(int(size))
        self.company_sizes = company_sizes
        # Generate sizes for every third-party
        _3rdParty_sizes = []
        for _ in self._3rdParty_IDs:
            x = np.linspace(0,self.d_size_3[2],10000)
            size = np.random.choice(x,
                                    size = 1,
                                    p = scipy.stats.halfnorm.pdf(x,self.d_size_3[0],self.d_size_3[1])/
                                        np.sum(scipy.stats.halfnorm.pdf(x,self.d_size_3[0],self.d_size_3[1])))[0]
            _3rdParty_sizes.append(int(size))
        self._3rdParty_sizes = _3rdParty_sizes
        # Generate sizes for every fourth-party
        _4thParty_sizes = []
        for _ in self._4thParty_IDs:
            x = np.linspace(0,self.d_size_4[2],10000)
            size = np.random.choice(x,
                                    size = 1,
                                    p = scipy.stats.halfnorm.pdf(x,self.d_size_4[0],self.d_size_4[1])/
                                        np.sum(scipy.stats.halfnorm.pdf(x,self.d_size_4[0],self.d_size_4[1])))[0]
            _4thParty_sizes.append(int(size))
        self._4thParty_sizes = _4thParty_sizes
        
    def generate_outgoing_connections(self):
        """
        Generate outgoing connections for companies and third-parties.
        """
        # Generate outgoing connections for every company
        company_outgoing_connections = []
        for _ in self.company_IDs:
            x = np.linspace(0,self.d_conn_c[2],10000)
            num_connections = np.random.choice(x,
                                               size = 1,
                                               p = scipy.stats.halfnorm.pdf(x,self.d_conn_c[0],self.d_conn_c[1])/
                                                   np.sum(scipy.stats.halfnorm.pdf(x,self.d_conn_c[0],self.d_conn_c[1])))[0]
            num_connections = int(np.ceil(num_connections))
            connections = list(set(np.random.choice(self._3rdParty_IDs,size=num_connections)))
            company_outgoing_connections.append(connections)
        self.company_outgoing_connections = company_outgoing_connections
        # Generate outgoing connections for every third-party
        _3rdParty_outgoing_connections = []
        for _ in self._3rdParty_IDs:
            x = np.linspace(0,self.d_conn_3[2],10000)
            num_connections = np.random.choice(x,
                                               size = 1,
                                               p = scipy.stats.halfnorm.pdf(x,self.d_conn_3[0],self.d_conn_3[1])/
                                                   np.sum(scipy.stats.halfnorm.pdf(x,self.d_conn_3[0],self.d_conn_3[1])))[0]
            num_connections = int(np.ceil(num_connections))
            connections = list(set(np.random.choice(self._4thParty_IDs,size=num_connections)))
            _3rdParty_outgoing_connections.append(connections)
        self._3rdParty_outgoing_connections = _3rdParty_outgoing_connections
        # No outgoing connections for every fourth-party
        _4thParty_outgoing_connections = [[] for _ in self._4thParty_IDs]
        self._4thParty_outgoing_connections = _4thParty_outgoing_connections
        
    def generate_incoming_connections(self):
        """
        Generate incoming connections for third-parties and fourth-parties.
        """
        # No incoming connections for every company
        company_incoming_connections = [[] for _ in self.company_IDs]
        self.company_incoming_connections = company_incoming_connections
        # Generate incoming connections for every third-party
        _3rdParty_incoming_connections = []
        for third_party in self._3rdParty_IDs:
            incoming_connections = []
            for i,company in enumerate(self.company_IDs):
                if third_party in self.company_outgoing_connections[i]:
                    incoming_connections.append(company)
            _3rdParty_incoming_connections.append(incoming_connections)
        self._3rdParty_incoming_connections = _3rdParty_incoming_connections
        # Generate incoming connections for every fourth-party
        _4thParty_incoming_connections = []
        for fourth_party in self._4thParty_IDs:
            incoming_connections = []
            for i,third_party in enumerate(self._3rdParty_IDs):
                if fourth_party in self._3rdParty_outgoing_connections[i]:
                    incoming_connections.append(third_party)
            _4thParty_incoming_connections.append(incoming_connections)
        self._4thParty_incoming_connections = _4thParty_incoming_connections
        
    def create_dataframe(self):
        """
        Generate dataframe of supply chain network
        """
        # Dataframe for every company
        company_df = pd.DataFrame([self.company_IDs,
                                   self.company_sizes,
                                   self.company_incoming_connections,
                                   self.company_outgoing_connections]).transpose()
        company_df["Type"] = "Company"
        company_df.columns = ["ID","Size","Incoming_Connections","Outgoing_Connections","Type"]
        company_df = company_df[["ID","Type","Size","Incoming_Connections","Outgoing_Connections"]]
        # Dataframe for every third-party
        _3rdParty_df = pd.DataFrame([self._3rdParty_IDs,
                                     self._3rdParty_sizes,
                                     self._3rdParty_incoming_connections,
                                     self._3rdParty_outgoing_connections]).transpose()
        _3rdParty_df["Type"] = "Third-Party"
        _3rdParty_df.columns = ["ID","Size","Incoming_Connections","Outgoing_Connections","Type"]
        _3rdParty_df = _3rdParty_df[["ID","Type","Size","Incoming_Connections","Outgoing_Connections"]]
        # Dataframe for every fourth-party
        _4thParty_df = pd.DataFrame([self._4thParty_IDs,
                                     self._4thParty_sizes,
                                     self._4thParty_incoming_connections,
                                     self._4thParty_outgoing_connections]).transpose()
        _4thParty_df["Type"] = "Fourth-Party"
        _4thParty_df.columns = ["ID","Size","Incoming_Connections","Outgoing_Connections","Type"]
        _4thParty_df = _4thParty_df[["ID","Type","Size","Incoming_Connections","Outgoing_Connections"]]
        # Overall supply chain network
        network = pd.concat([company_df,_3rdParty_df,_4thParty_df])
        network["All_Connections"] = (network["Incoming_Connections"]+network["Outgoing_Connections"]).apply(lambda x: list(set(x)))
        network = network.reset_index(drop=True)
        self.network = network
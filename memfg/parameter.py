class Parameter:
    '''
    Parameter class.
    
    ----
    name : str
    
    value : double, or numpy.ndarray
    
    '''
    def __init__(self,name=None,value=None):
        self.__name = name
        self.__value = value
        
    @property
    def name(self)
        return self.__name
    
    @property
    def value(self)
        return self.__value
    
Parameter.to_array = Parameter.value
    
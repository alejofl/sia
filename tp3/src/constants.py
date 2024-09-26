class Constants:
    _instance = None

    def __new__(cls, **kwargs):
        if cls._instance is None:
            cls._instance = super(Constants, cls).__new__(cls)
            cls._instance._initialize(**kwargs)
        return cls._instance

    def _initialize(self, **kwargs):
        self.beta = kwargs.get("beta", 1)
        self.epsilon = kwargs.get("epsilon", 1e-5)
        self.seed = kwargs.get("seed", None)
        self.maxEpochs = kwargs.get("maxEpochs", 1000)
        self.learningRate = kwargs.get("learningRate", 0.1)

    @staticmethod
    def getInstance():
        if Constants._instance is None:
            raise ValueError("Singleton class hasn't been instantiated")
        return Constants._instance

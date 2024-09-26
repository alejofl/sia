class Constants:
    _instance = None

    def __new__(cls, beta):
        if cls._instance is None:
            cls._instance = super(Constants, cls).__new__(cls)
            cls._instance._initialize(beta)
        return cls._instance

    def _initialize(self, beta):
        self.beta = beta

    @staticmethod
    def getInstance():
        if Constants._instance is None:
            raise ValueError("Singleton class hasn't been instantiated")
        return Constants._instance

from sqlalchemy import create_engine


class DatabaseClient:
    def __init__(self):
        self.cnx = create_engine("mysql+pymysql://gold:gold@localhost/gold")

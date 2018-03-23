from __future__ import print_function, unicode_literals
from bosonnlp import BosonNLP
import pymysql

nlp = BosonNLP('Lj2kJwo0.24552.P6x3Uun0UnGv')

conn = pymysql.connect(host='localhost',
                             user='root',
                             passwd='LUKEHE051308',
                             db='spider163',
                             charset='utf8',
                             )

#example
cursor = conn.cursor()
sql = "select comment_all from songinfo where song_id = 28302393"
cursor.execute(sql)
comment = cursor.fetchone()
result = nlp.sentiment(comment)
print(result)
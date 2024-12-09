#--------------------------------------------------------------------------共用
#使用模型 名稱與路徑 
load_directories={
        'bert':"Model\\bert_emb",
        'tao':"Model\\tao",
        'bge':"Model\\bge-large-zh-v1.5"
    }
#資料庫路徑
Database_root = 'Database.csv'

#--------------------------------------------------------------------------檔案生成
#資料夾路徑 裡面放要轉換的excel 以第一欄為問題 第二欄為答案
data_directories = r"file_to_database"

#資料庫的欄位名稱
fieldnames = ['num','QA', 'Category', 'Filename', 'bert_emb', 'tao_emb', 'bge_emb']

#餘弦門檻 (當新的資料 跟資料庫舊有資料餘閒值高於這個值,就不存進資料庫=>避免重複資料)
sim=0.92 

#儲存相似資料檔案資料夾路徑
Similar_questions_root="Similar_questions"

#相似資料欄位名稱
Similar_question_col=['題目', '與資料庫相似題目', 'Category', 'Filename']


#--------------------------------------------------------------------------RAG
#使用的模型 輸入vicuna/gpt
use_model="vicuna"

#找出最相關資料的筆數 會影響模型的max_token 找越多筆 模型過載機率越大
result_limit=3

#儲存空間上限
maxBytes=5*1024*1024
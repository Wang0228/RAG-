import pandas as pd
import csv
import faiss
import torch
import os
import numpy as np
import glob
from transformers import AutoTokenizer, AutoModel
import config
import re

def get_max_num_from_db():  # 從資料庫獲取目前最大的num
    if Database_QA:
        return max(int(record.get('num', 0)) for record in Database_QA)
    return 0
def get_exist_QA_emb(filename):#載入現有QA
    global Database_QA,num
    print("載入資料庫資料")
    df=pd.read_csv(filename)
    Database_QA=df[['num','QA', 'Category', 'Filename']].to_dict(orient='records')
    for name in config.load_directories.keys():
        emb_col=f'{name}_emb'
        embedding_data=np.stack(df[emb_col].apply(eval).to_numpy())
        D=embedding_data.shape[1]
        model[f'{name}_index']=faiss.IndexFlatIP(D)
        model[f'{name}_index'].add(embedding_data)
    num = get_max_num_from_db()
    print("載入完成")

def model_ready(load_directories):#模型載入
    global model
    print("模型載入")
    for name,directory in config.load_directories.items():
        model[f'{name}_tokenizer']=AutoTokenizer.from_pretrained(directory)
        model[f'{name}_model']=AutoModel.from_pretrained(directory)
    print("載入完成")

def data_generate(file_path,writer):#生成儲存資料
    global database_exist
    global Database_QA,num

    df_excel = pd.read_excel(file_path)
    for index, row in df_excel.iterrows():   
        question = row['Q']
        answer = row['A']
        category = row['Category']
        filename = row['Filename']
        QA = f"{question} {answer}"
        if not(database_exist):#如果資料庫是新創的
            bert_emb = format_to_database(get_embedding(model['bert_model'], model['bert_tokenizer'], QA))
            tao_emb = format_to_database(get_embedding(model['tao_model'], model['tao_tokenizer'], QA))
            bge_emb = format_to_database(get_embedding(model['bge_model'], model['bge_tokenizer'], QA))
            num += 1
            writer.writerow({'num':num,'QA': QA, 'Category': category, 'Filename': filename, 'bert_emb': bert_emb, 'tao_emb': tao_emb, 'bge_emb': bge_emb})
            get_exist_QA_emb(config.Database_root)
            database_exist=1
        else:#舊有資料庫
            bert_emb = get_embedding(model['bert_model'], model['bert_tokenizer'], QA)
            tao_emb = get_embedding(model['tao_model'], model['tao_tokenizer'], QA)
            bge_emb = get_embedding(model['bge_model'], model['bge_tokenizer'], QA)

            distances_bert, indices_bert = model['bert_index'].search(bert_emb, 1)
            distances_tao, indices_tao = model['tao_index'].search(tao_emb,1)
            distances_bge, indices_bge= model['bge_index'].search(bge_emb,1)

            #餘閒值
            # print(str(distances_bert[0][0]))
            # print(str(distances_tao[0][0]))
            # print(str(distances_bge[0][0]))

            distances_over_threshold = sum([d[0][0] > config.sim for d in [distances_bert, distances_tao, distances_bge]])#只要有其中兩筆餘弦大於門檻就不處理
            if distances_over_threshold <= 2:
                model['bert_index'].add(bert_emb)
                model['tao_index'].add(tao_emb)
                model['bge_index'].add(bge_emb)
                bert_emb_str = format_to_database(bert_emb)
                tao_emb_str = format_to_database(tao_emb)
                bge_emb_str = format_to_database(bge_emb)
                Database_QA.append({'QA': QA, 'Category': category, 'Filename': filename})
                num += 1
                writer.writerow({'num':num,'QA': QA, 'Category': category, 'Filename': filename, 'bert_emb': bert_emb_str, 'tao_emb': tao_emb_str, 'bge_emb': bge_emb_str})
            else:
                similar_record = Database_QA[int(indices_tao[0][0])]
                print(f"----------------------\n\n{QA} \n\n與 資料庫問題\n\n{similar_record['QA']} \n\n(Category: {similar_record['Category']}, Filename: {similar_record['Filename']}) 相似資料")
                record_data = {
                    '題目': QA,
                    '與資料庫相似題目': similar_record['QA'],
                    'Category': similar_record['Category'],
                    'Filename': similar_record['Filename']
                }
                record_similar_questions(record_data)

def create_similar_question_file(folder):#創建相似資料檔
    max_number = 0
    pattern = r"Similar_question(\d+).xlsx"
    for filename in os.listdir(folder):
        match = re.search(pattern, filename)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    new_filename = f"Similar_question{str(max_number + 1).zfill(2)}.xlsx"
    print(f"創建儲存相似問題檔案,檔名:{new_filename}")
    new_file_path = os.path.join(folder, new_filename)
    df_new = pd.DataFrame(columns=config.Similar_question_col)
    df_new.to_excel(new_file_path, index=False)
    print("創建完成")
    return new_filename

def record_similar_questions(data):#儲存相似資料
    global similar_question_file_path
    file_path = config.Similar_questions_root + "/" + similar_question_file_path
    df_existing = pd.read_excel(file_path)
    df_new = pd.DataFrame([data], columns=config.Similar_question_col)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_excel(file_path, index=False)
        

def get_embedding(model,tokenizer,text):#取得向量
    inputs=tokenizer(text,return_tensors='pt')
    outputs=model(**inputs)
    embeddings = outputs.last_hidden_state.mean(1)
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1).detach().numpy()
    return normalized_embeddings
    
def format_to_database(normalized_embeddings):
    embeddings_str = ",".join(["{:.18f}".format(x) for x in normalized_embeddings[0]])
    return embeddings_str


model={}#用來儲存分詞器及模型
Database_QA=[]#現有的QA
database_exist=1
similar_question_file_path=""
num=0
if not os.path.exists(config.Database_root):#沒有資料庫 創建一個
    print("創建新資料庫")
    with open(config.Database_root, 'w', newline='', encoding='utf-8') as file:
        database_exist=0
        writer = csv.DictWriter(file, fieldnames=config.fieldnames)
        writer.writeheader()
    print("創建完成")
else:
    get_exist_QA_emb(config.Database_root)#取得現有QA
similar_question_file_path=create_similar_question_file(config.Similar_questions_root)#創建儲存相似資料檔案
model_ready(config.load_directories)#模型載入

with open(config.Database_root, 'a', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=config.fieldnames)
    xlsx_files = glob.glob(os.path.join(config.data_directories, "*.xlsx"))
    for file in xlsx_files:
        print("正在處理檔案:", file)
        data_generate(file,writer)
print(f"儲存重複資料檔名:{similar_question_file_path}(位於Similar_questions資料夾)")
print("Done!")
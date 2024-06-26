import class_0623

def main():
    class_0623.plot_cost_function()

if __name__ == "__main__":
    main()

    # # 함수 내부에 try 구문을 사용하는 것은 예외 처리(Exception Handling)를 위해서입니다.
# # 코드가 실행되는 동안 발생할 수 있는 오류(예외)를 잡아서 프로그램이 비정상적으로 종료되지 않도록 하고,
# # 적절한 처리를 할 수 있게 합니다. 이를 통해 코드의 안정성과 신뢰성을 높일 수 있습니다.
# # try 구문을 사용하는 경우를 몇 가지 예로 설명하겠습니다.
# 
# 
# 
# 
# # 1. 예외 발생 가능성이 있는 코드 실행
# # 예를 들어, 리스트의 인덱스를 접근할 때, 해당 인덱스가 리스트 범위를 벗어날 수 있는 경우 IndexError가 발생할 수 있습니다.
# # 이런 상황에서 try 구문을 사용하면 예외를 처리하고 프로그램이 정상적으로 계속 실행되도록 할 수 있습니다.
# 
# def calc(list_data):
#     sum = 0
# 
#     try:
#         sum = list_data[0] + list_data[1] + list_data[2]
# 
#         if sum < 0:
#             raise Exception("sum is minus")
# 
#     except IndexError as err:
#         print(str(err))
#     except Exception as err:
#         print(str(err))
#     finally:
#         print(sum)
# 
# calc([1,2,-100])
# 
# 
# 
# # 2. 파일 I/O 작업
# # 파일을 읽거나 쓸 때 파일이 존재하지 않거나 읽기/쓰기 권한이 없으면 예외가 발생할 수 있습니다.
# # 이 경우 try 구문을 사용하여 예외를 처리할 수 있습니다.
# 
# def read_file(file_path):
#     try:
#         with open(file_path, 'r') as file:
#             content = file.read()
#             print(content)
#     except FileNotFoundError as e:
#         print("File not found:", e)
#     except IOError as e:
#         print("I/O error:", e)
# 
# read_file('non_existent_file.txt')
# 
# # 3. 네트워크 통신
# # 네트워크 통신 중에는 연결 오류, 타임아웃 등 다양한 예외가 발생할 수 있습니다.
# # 이런 예외를 try 구문을 통해 처리할 수 있습니다.
# 
# import requests
# 
# def fetch_data(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         data = response.json()
#         print(data)
#     except requests.exceptions.HTTPError as e:
#         print("HTTP error:", e)
#     except requests.exceptions.ConnectionError as e:
#         print("Connection error:", e)
#     except requests.exceptions.Timeout as e:
#         print("Timeout error:", e)
#     except requests.exceptions.RequestException as e:
#         print("Request exception:", e)
# 
# fetch_data('https://api.example.com/data')
# 
# # 4. 데이터 변환 및 처리
# # 사용자로부터 입력받은 데이터를 처리할 때, 데이터 형식이 예상과 다를 수 있습니다.
# # 예를 들어, 문자열을 숫자로 변환할 때 ValueError가 발생할 수 있습니다.
# # 이런 경우 try 구문을 사용하여 예외를 처리할 수 있습니다.
# 
# def convert_to_int(value):
#     try:
#         return int(value)
#     except ValueError as e:
#         print("Conversion error:", e)
#         return None
# 
# print(convert_to_int("123"))  # 정상적인 경우
# print(convert_to_int("abc"))  # 예외 발생 경우

#
# import spacy
# from spacy.tokens import Span
# import networkx as nx
# import matplotlib.pyplot as plt
#
#
# # Spacy 모델 로드 (영어)
# nlp = spacy.load("en_core_web_sm")
#
# # 처리할 문장
# sentence = "Apple is looking at buying U.K. startup for $1 billion."
#
# # 사용자 정의 개체 인식 규칙 추가
# def add_custom_entity_rule(doc):
#     new_ents = []
#     for ent in doc.ents:
#         if ent.text == "U.K." or ent.text == "U.K. startup":
#             new_ent = Span(doc, ent.start, ent.end, label="ORG")
#             new_ents.append(new_ent)
#         else:
#             new_ents.append(ent)
#     doc.ents = new_ents
#     return doc
#
# # 사용자 정의 구성 요소 등록
# @spacy.Language.component("custom_entity_rule")
# def custom_entity_rule_component(doc):
#     return add_custom_entity_rule(doc)
#
# # 사용자 정의 구성 요소를 파이프라인에 추가
# nlp.add_pipe("custom_entity_rule", after="ner")
#
# # 문장 처리
# doc = nlp(sentence)
#
# # 개체 인식 결과 출력
# for ent in doc.ents:
#     print(ent.text, "-", ent.label_)

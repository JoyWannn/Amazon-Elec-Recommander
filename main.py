# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import http.server
import math
import os
import socketserver
import seaborn as sns
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 40)
pd.set_option('display.width', 300)

class Config:
    data_path = r'C:\Users\08156\recommander_final'
    fn_product_path = os.path.join(data_path,'ElecNew.csv') #anazon電子產品資料
    fn_comment_path = os.path.join(data_path,'Comments.csv') #amazon電子產品的評分資料
    web_port = 8001  # 網頁伺服器的埠號
    web_mode = True # True: 啟用網頁伺服器; False: 在文字介面直接執行程式

class ItemDatabase:

    def __init__(self):
        self.items = None  # 稍後載入
        self.load()  # 載入物品資料庫

    def load(self):
        fn_product_info = Config.fn_product_path

        if os.path.exists(fn_product_info):
            try:
                print('從', fn_product_info, '載入商品資料。')
                self.items = pd.read_csv(fn_product_info)
                print('目前有 {:d} 筆物品資料在資料庫中。'.format(len(self.items)))
            except Exception:
                print('嚴重錯誤：無法讀取既有的檔案。程式結束。', fn_product_info)
                exit()
        else:
            print('找不到', fn_product_info, '。程式結束')
            exit()
    def get_product_by_id(self, item_id):
        return self.items[self.items['asin'] == item_id].to_dict('records')[0]

    def get_product_by_id_list(self, item_ids):
        # 回傳內容為一個清單，清單中的每一個元素是一個字典，包含一筆物品的所有資料。
        result = []
        for item_id in item_ids:
            result.append(self.get_product_by_id(item_id))
        return result

    def get_number_of_items(self):
        # 取得資料庫裡面所有物品的總數
        return len(self.items)

class UserHistory:
    def __init__(self):
        self.fn_user_rating = Config.fn_comment_path
        self.fn_user_rating_configurable = False
        self.user_rating = None
        self.user_id = None

        self.load_or_create()
        while self.user_id is None:
            # 如果用戶沒有登入，則會一直重複詢問
            self.sign_in_or_sign_up()
    def load_or_create(self, allow_slicing=False):
        # 如果檔案存在，則嘗試讀取檔案
        if os.path.exists(self.fn_user_rating):
            try:
                print('讀取用戶評分資料:', self.fn_user_rating)
                self.user_ratings = pd.read_csv(self.fn_user_rating)
                return
            except Exception:
                print('嚴重錯誤：無法讀取', self.fn_user_rating, '。程式結束。')
                exit()
        #如果檔案不存在則載入評分資料或建立空資料庫
        else:
            print('找不到舊的用戶評分紀錄檔', self.fn_user_rating)
            response = input('要將 Amazon 的用戶評分資料載入到資料庫中嗎? (Y/n)')
            if response.lower() not in ('n', 'no'):
                # 如果用戶沒有不同意載入 Amazon 評分紀錄，就載入
                self.copy_from_amazonElec()
            else:
                # 如果不載入 Amazon，就建立一個空的資料庫
                print('不載入 Amazon Comment。重新建立一個存放未來用戶評分的全新容器。')
                self.user_ratings = pd.DataFrame({'reviewerID': pd.Series(dtype='int64'),
                                                  'asin': pd.Series(dtype='int64'),
                                                  'overall': pd.Series(dtype='float64')})

        if allow_slicing and len(self.user_ratings) > 0:
            response = input('如果要只保留最活躍的前幾個用戶和前幾名最熱門的商品，請輸入一個數量，或是直接按enter跳過:')
            try:
                # 如果輸入的是一個正整數，就只保留該數量的用戶和電影
                if int(response) > 0:
                    self.keep_only_active_users_and_popular_electrical(int(response))
            except ValueError:
                pass

    def copy_from_amazonElec(self):
        # 若有需要，可以採用 Amazon 的用戶評分紀錄作為開始，避免資料庫一片空白

        print('從', Config.fn_comment_path, '載入 Amazon 的用戶評分資料。')
        with open('fn_comment_path', 'r') as f:
            self.user_ratings = pd.read_csv(f, encoding='utf8', sep=',')

        self.user_ratings = self.user_ratings.loc[:, ['reviewerID', 'asin', 'overall']]
        print('已經載入 {:d} 筆評分資料到資料庫中。'.format(len(self.user_ratings)))

    def keep_only_active_users_and_popular_electrical(self, k, sample_fraction=0.5):
        # 在self.user_ratings裡面，僅保留有最多評分的k個用戶，以及這些用戶給最多評分的k個電影，並隨機抽樣某個比例的評分紀錄
        # 這個可以用來縮減資料量，加速實驗
        print('只保留最活躍的', k, '個用戶和最熱門的', k, '個商品。並保留百分之', sample_fraction * 100, '的評分紀錄。')

        # 取得最活躍的k個用戶
        active_users = self.user_ratings.groupby('reviewerID').count().nlargest(k, 'asin').index

        # 取得這些活躍用戶的所有評分
        self.user_ratings = self.user_ratings[self.user_ratings['reviewerID'].isin(active_users)]

        # 取得這些活躍用戶給最多評分的k個熱門電影
        popular_items = self.user_ratings.groupby('asin').count().nlargest(k, 'reviewerID').index

        # 只留下這些活躍用戶給這些熱門電影的評分，並隨機刪除一些評分紀錄，保留之後推薦的空間
        selector = (self.user_ratings['reviewerID'].isin(active_users)) & (self.user_ratings['asin'].isin(popular_items))
        self.user_ratings = self.user_ratings[selector].sample(frac=sample_fraction)

    def show_login_message(self):
        print('你登入的用戶編號是 {:d}。'.format(self.user_id))
        print('你有 {:d} 筆評分紀錄存在資料庫中。'.format(self.get_rating_count()))
        if self.get_rating_count() == 0:
            print('你可能是新用戶')

    def sign_in_or_sign_up(self):
        response = input('請輸入你想使用的純數字用戶編號， \n'
                         '你也可以輸入 new 創造新用戶，或是輸入 active 使用有最多評分的帳戶：')

        # 建立新用戶帳號 #lower是返回值
        if response.lower() == 'new':
            max_id = self.get_max_user_id()
            if max_id is None:
                self.user_id = 0
                print('目前資料庫中沒有任何用戶，因此你的用戶編號是 0。')
            else:
                self.user_id = max_id + 1
                print('已經為你分配一個全新的用戶編號 {:d}。'.format(self.user_id))

        # 使用最活躍的用戶帳號
        elif response.lower() == 'active':
            active_id = self.get_most_active_user_id()
            if active_id is None:
                self.user_id = 0
                print('目前資料庫中沒有任何用戶，因此你的用戶編號是 0。')
            else:
                self.user_id = active_id
                print('你的用戶編號是 {:d}。'.format(self.user_id))

        # 用戶指定編號，若不存在，就新增該用戶
        elif response.isdigit():
            self.user_id = int(response)
            self.show_login_message()

        # 不允許其他狀況
        else:
            print('你的輸入錯誤，請使用純數字用戶編號，或是輸入 new 或 active。')
            return False

        return True

    def save_to_disk(self):
        print('正在儲存用戶評分紀錄到檔案', self.fn_user_rating)
        self.user_ratings.to_csv(self.fn_user_rating, index=False)

    def set_rating(self, item, rating):
        # 新增或更改目前用戶對特定物品的評分

        # 檢查該用戶對於該物品是否已經有評分
        selector = (self.user_ratings['reviewerID'] == self.user_id) & \
                   (self.user_ratings['asin'] == item['asin'])
        series = self.user_ratings.loc[selector, 'overall']

        # 之前沒評分，直接新增
        if len(series) == 0:
            print('新增用戶', self.user_id, '對於物品', item['asin'], '的評分:', rating)
            data_dict = {'reviewerID': self.user_id, 'asin': item['asin'], 'overall': rating}
            new_row = pd.DataFrame(data_dict, index=[-1])  # 索引值-1可以隨意給，因為等一下要ignore_index忽略它
            self.user_ratings = pd.concat([self.user_ratings, new_row], ignore_index=True)

        # 之前有一個評分，更改評分
        elif len(series) == 1:
            print('更新用戶', self.user_id, '對於物品', item['asin'], '的評分:', rating)
            self.user_ratings.loc[selector, 'overall'] = rating

        # 其他狀況
        else:
            print('警告：用戶', self.user_id, '對於物品', item['asin'], '有超過一筆評分，放棄更新評分')

    def get_top_rated_items(self, k):
        # 取得用戶最喜歡的n個物品，回傳前k個amazon商品清單

        # 取得用戶的評分
        user_ratings = self.user_ratings[self.user_ratings['reviewerID'] == self.user_id]

        # 取得用戶評分最高的n個物品
        user_ratings = user_ratings.sort_values(by='rating', ascending=False)

        # 僅回傳商品asin清單
        return user_ratings['asin'].head(k).tolist()

    def get_user_rating(self, item):
        # 取得用戶對於特定物品的評分

        selector = (self.user_ratings['reviewerID'] == self.user_id) & (self.user_ratings['asin'] == item['asin'])
        series = self.user_ratings.loc[selector, 'overall']

        if len(series) == 0:
            return None
        elif len(series) == 1:
            return series.item()
        else:
            print('警告：用戶', self.user_id, '對於物品', item['asin'], '有超過一筆評分')
            return None

    def get_rating_count(self):
        # 取得當前用戶的評分筆數
        return len(self.user_ratings[self.user_ratings['reviewerID'] == self.user_id])

    def get_max_user_id(self):
        # 取得最大的用戶編號。加一之後可以作為新用戶的編號
        if len(self.user_ratings) == 0:
            return None
        else:
            return self.user_ratings['reviewerID'].max()

    def get_most_active_user_id(self):
        # 取得有最多評分紀錄的用戶
        ratings_per_user = self.user_ratings.groupby('reviewerID')['overall'] #這邊的語法是啥意思
        count_ratings_per_user = ratings_per_user.count()

        if len(count_ratings_per_user) == 0:
            return None

        # 取得用戶編號最大值的索引值
        user_id_with_most_ratings = count_ratings_per_user.idxmax()
        print('最活躍的用戶編號是', user_id_with_most_ratings, '，具有',
              len(self.user_ratings[self.user_ratings['reviewerID'] == user_id_with_most_ratings]), '個評分紀錄。')

        return user_id_with_most_ratings

    def get_number_of_users(self):
        # 取得用戶數量
        return len(self.user_ratings['reviewerID'].unique())

    #做到這邊
    def purge_ratings_for_unlisted_product(self, item_database):
        # 刪除用戶評分紀錄中，沒有在item_database中的電影的評分紀錄
        # 由於電影資料在清理與整合過程中，可能會刪除一些原有的資料，因此對應的評分資料也需要刪除

        print('刪除用戶評分紀錄中，沒有在item_database中的電影的評分紀錄，目前有', len(self.user_ratings), '筆評分紀錄。')
        # 取得所有的asin
        product_asin = item_database.items['asin'].tolist()

        # 刪除用戶評分紀錄中，沒有在item_database中的電影的評分紀錄
        self.user_ratings = self.user_ratings[self.user_ratings['asin'].isin(product_asin)]
        print('刪除後的用戶評分紀錄數量：', len(self.user_ratings))

    def get_number_of_ratings(self):
        # 取得評分紀錄數量
        return len(self.user_ratings)

    def get_user_genre_preference(self, item_database, user_id=None):
        # 取得用戶評分過的電影的類型分佈，回傳一個dict，key是類型，value是該類型的電影的評分總和，再正規化
        # 用來理解某個用戶對於電影類型的喜好程度

        if user_id is None:
            user_id = self.user_id

        # 取得用戶的評分紀錄
        selector = (self.user_ratings['reviewerID'] == user_id)
        user_ratings = self.user_ratings.loc[selector, ['asin', 'overall']]

        #這個待改
        # 取得所有電影類別欄位
        col_genres = item_database.items.columns[
            item_database.items.columns.str.startswith(item_database.genre_col_prefix)]

        #這個待改
        # 取得用戶評分過的電影的類型分佈
        items = item_database.items[['asin'] + col_genres.tolist()]
        items = items.merge(user_ratings, on='asin', how='inner')

        # 這邊待改
        # 計算用戶評分過的電影的類型分佈
        user_genre_preference = {}
        for genre in col_genres:
            user_genre_preference[genre] = items[items[genre] == 1]['overall'].sum()

        #這邊待改
        # 進行正規化，讓偏好分數總合為1
        total = sum(user_genre_preference.values())
        if total > 0:
            for genre in user_genre_preference:
                user_genre_preference[genre] /= total

        return user_genre_preference

    def get_most_focused_users(self, item_database):
        # 取得最專注的用戶，也就是評分過的電影的類型分佈最集中的用戶，回傳一個list，裡面是用戶編號，由專注度由高到低排序
        # 方便找到那些比較適合做為實驗範例的用戶

        print('取得最專注的用戶，也就是評分過的電影的類型分佈最集中的用戶。')

        # 依照用戶評分數量排序，只看評分數量較少的用戶
        ratings_per_user = self.user_ratings.groupby('reviewerID')['overall']
        count_ratings_per_user = ratings_per_user.count()
        count_ratings_per_user.sort_values(ascending=True, inplace=True)
        user_ids = count_ratings_per_user.index[len(count_ratings_per_user) // 10:]

        # 逐一處理每個用戶，取得該用戶的評分過的電影的類型分佈
        user_genre_preferences = []
        for user_id in user_ids:
            user_genre_preference = self.get_user_genre_preference(item_database, user_id)
            user_genre_preference['reviewerID'] = user_id
            user_genre_preferences.append(user_genre_preference)

        # 將user_genre_preferences轉換成DataFrame，並將索引設定為userId
        user_genre_preference_dataframe = pd.DataFrame(user_genre_preferences)
        user_genre_preference_dataframe.set_index('reviewerID', inplace=True)

        #待改
        def entropy(row):
            # 依照單一用戶對於電影類型偏好的分布，用entropy計算impurity
            return -sum([p * math.log(p) if p > 0 else 0 for p in row if not math.isnan(p)])

        # 依照表格內容，計算impurity，並將結果加入到user_genre_preference_dataframe
        user_genre_preference_dataframe['entropy'] = user_genre_preference_dataframe.apply(lambda row: entropy(row),
                                                                                           axis=1)

        # 依照entropy排序
        user_genre_preference_dataframe.sort_values(by='entropy', ascending=True, inplace=True)

        # 全部乘上100再轉成整數，方便觀看
        user_genre_preference_dataframe *= 100
        user_genre_preference_dataframe = user_genre_preference_dataframe.astype(int)

        print('依照entropy排序，類別偏好最集中的前20名的用戶的電影類型偏好分布：')
        print(user_genre_preference_dataframe.head(20))

        # 回傳排序後的用戶清單
        return user_genre_preference_dataframe.index.tolist()


class RecommenderSim:
    def __init__(self, num_recommend=20):
        print('啟動推薦系統模擬器。')
        self.num_recommend = num_recommend  # 推薦的數量
        self.last_recommendation = None  # 最後一次的推薦結果
        self.online = True  # 是否開機運作中，若為False則終止本系統

        self.user_history = UserHistory()  # 使用者評分紀錄，以當前用戶身分進行新增、查詢、修改
        self.item_database = ItemDatabase()  # 物品資料庫，唯讀

        self.user_history.purge_ratings_for_unlisted_product(self.item_database)  # 清除資料庫中沒有的電影評分

        # self.user_history.get_most_focused_users(self.item_database)  # 取得最專注的用戶

    def update_item_features(self):
        print('正在更新物品內容特徵資料庫')
        pass

    def refresh_recommendation(self, method='popular'):
        # 重新建立推薦清單，推薦算法的進入點
        print('系統正在建立你可能喜歡的物品清單。')

        if method == 'user_based':
            print('使用基於用戶的協同過濾推薦物品')
            self.last_recommendation = Algorithm.user_based_collaborative_filtering(
                self.user_history.user_ratings, self.user_history.user_id, self.num_recommend)
            self.last_recommendation = self.item_database.get_product_by_id_list(self.last_recommendation.index)

        if len(self.last_recommendation) == 0 or method == 'popular':
            print('推薦熱門物品')
            self.last_recommendation = Algorithm.popularity_based_recommendation(
                self.user_history.user_ratings, self.user_history.user_id, self.num_recommend)
            self.last_recommendation = self.item_database.get_product_by_id_list(self.last_recommendation.index)

    def show_items(self, k_favorite=20):
        # 取得用戶對於不同電影類型的偏好，並將結果儲存到 genre_preference 變數中，該變數為一個字典，其中的鍵為電影類型，值為偏好
        genre_preference = rec.user_history.get_user_genre_preference(rec.item_database)
        # 將 genre_preference 變數中的商品類型依照偏好由大到小排序
        genre_preference = sorted(genre_preference.items(), key=lambda x: x[1], reverse=True)
        print('你最喜歡的電影類型是:')
        print('電影類型 偏好')
        print('-' * 60)
        for genre, preference in genre_preference:
            print(genre, '{:.0%}'.format(preference))
        print('-' * 60)

        # 基於推薦清單，以適當形式呈現給用戶，如果用戶對該物品有評分，一併顯示
        print('你最喜歡的物品是:')
        print('編號 評分 物品標題')

        # 依照用戶的評分排序，評分高的在前面，取得用戶最喜歡的前k個物品
        favorite_asin = self.user_history.get_top_rated_items(k_favorite)
        favorite_items = self.item_database.get_product_by_id_list(favorite_asin)

        # 顯示用戶最喜歡的物品
        print('-' * 60)
        for serial, item in enumerate(favorite_items):
            rating = self.user_history.get_user_rating(item)
            rating = '[?.?]' if rating is None else '[{:.1f}]'.format(rating)
            print('{:d}.'.format(serial), rating, item['title'], item['asin'])
        print('-' * 60)

        print('推薦以下物品給你:')
        print('編號 評分 物品標題')

        # 顯示推薦清單
        print('-' * 60)
        for serial, item in enumerate(self.last_recommendation):
            rating = self.user_history.get_user_rating(item)
            rating = '[?.?]' if rating is None else '[{:.1f}]'.format(rating)
            print('{:d}.'.format(serial), rating, item['title'], item['asin'])
        print('-' * 60)

    def get_user_feedback(self):
        # 反覆詢問並檢查使用者對於物品的評分回應

        while True:
            response = input('請輸入你喜歡的物品編號，或是輸入 s 停止回饋並更新推薦清單，或是輸入 u 切換用戶，或是輸入 q 離開系統。')

            if response.lower() == 's':
                return 'stop_feedback'
            if response.lower() == 'u':
                return 'switch_user'
            if response.lower() == 'q':
                return 'quit'

            try:
                if 0 <= int(response) <= len(self.last_recommendation) - 1:
                    serial = int(response)
                    print('你選擇了 #{:d} 物品。'.format(serial))
                else:
                    print('請輸入介於 0 到 {:d} 的編號。'.format(len(self.last_recommendation) - 1))
                    continue
            except ValueError:
                print('無法辨識你的輸入。')
                continue

            response = input('請輸入你對於編號 #{:d} 物品的評分，0 到 5 分：'.format(serial))
            try:
                if 0 <= int(response) <= 5:
                    rating = int(response)
                    print('你對於編號 #{:d} 物品給了 {:d} 分。'.format(serial, rating))
                    return serial, rating
                else:
                    print('請輸入介於 0 到 5 的評分。')
                    continue
            except ValueError:
                print('無法辨識你的輸入。')
                continue

    def user_feedback_loop(self):
        # 反覆詢問用戶，嘗試取得多組物品的評分，直到用戶停止回饋

        while True:
            response = self.get_user_feedback()  # 請用戶選擇一個物品並給予評分

            if response == 'stop_feedback':
                # 用戶選擇暫停輸入，看更新的推薦結果
                return
            elif response == 'switch_user':
                # 用戶選擇切換用戶
                self.user_history.user_id = int(input('請輸入用戶編號：'))
                self.user_history.show_login_message()
                self.last_recommendation = None
                break
            elif response == 'quit':
                # 用戶選擇終止本系統
                self.online = False
                break
            else:
                # 已經取得用戶對於一個特定物品的評分，更新資料庫
                serial, rating = response
                self.user_history.set_rating(self.last_recommendation[serial], rating)


    def main_loop(self):
        # 在非Web模式下，程式的主要進入點，控制最高層次的邏輯順序

        while self.online:
            print()
            self.update_item_features()  # 更新物品特徵資料庫
            self.refresh_recommendation(method='user_based')   # 更新推薦清單
            self.show_items()  # 顯示推薦清單給用戶
            self.user_feedback_loop()  # 反覆取得用戶評分或指令
            self.update_user_profile()  # 更新用戶偏好模型

        # 離開之前，把評分紀錄存檔。若非正常終止程式，則評分紀錄不會存檔
        self.user_history.save_to_disk()
        print('系統正常結束。')

class RequestHandler(http.server.BaseHTTPRequestHandler):
    # 用來處理HTTP請求的類別，並覆寫了一些方法，以便處理我們的請求，並回應我們的網頁

    KEY_RATING_PREFIX = 'rating_'  # 用戶對於物品的評分
    KEY_SWITCH_USER = 'switch_user'  # 用戶選擇切換用戶
    #TMDB_IMG_BASE_URL = 'https://image.tmdb.org/t/p/w440_and_h660_face'
    # 用戶對於物品的評分, float32, 0~5, 0.5為單位
    VALID_RATINGS = (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)

    def show_page_title(self):
        # 顯示網頁標題，以及設定網頁的編碼為 UTF-8，並取消網頁的快取機制，以便即時更新網頁內容，而不是使用快取的內容
        self.wfile.write('<html><head>'.encode('utf-8'))
        self.wfile.write(
            '<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />'.encode('utf-8'))
        self.wfile.write('<title>Amazon商品推薦實驗平台</title>'.encode('utf-8'))

        # 設定網頁的字型為微軟正黑體，並設定表格的樣式，以便網頁看起來更好看，也更容易閱讀，而不是使用瀏覽器預設的樣式
        self.wfile.write('<style>'.encode('utf-8'))
        self.wfile.write('body { font-family: "微軟正黑體"; }'.encode('utf-8'))
        self.wfile.write('h1 { font-size: 1.5em; }'.encode('utf-8'))
        self.wfile.write('h2 { font-size: 1.2em; }'.encode('utf-8'))
        self.wfile.write('h3 { font-size: 1.0em; }'.encode('utf-8'))
        self.wfile.write('table { border-collapse: collapse; }'.encode('utf-8'))
        self.wfile.write('table, th, td { border: 1px solid black; }'.encode('utf-8'))
        self.wfile.write('th, td { padding: 5px; }'.encode('utf-8'))
        self.wfile.write('th { text-align: left; }'.encode('utf-8'))
        self.wfile.write('</style>'.encode('utf-8'))

        # 設定網頁的圖示為電影的圖示
        self.wfile.write('</head><body>'.encode('utf-8'))

        # 顯示網頁的標題
        self.wfile.write('<h1>Amazon商品推薦實驗平台</h1>'.encode('utf-8'))

        # 顯示目前資料庫中的資料數量
        n_users = rec.user_history.get_number_of_users()
        n_movies = rec.item_database.get_number_of_items()
        n_ratings = rec.user_history.get_number_of_ratings()
        self.wfile.write('<p>目前資料庫中有 {:d} 位用戶，{:d} 部電影，{:d} 筆評分紀錄。</p>'.format(
            n_users, n_movies, n_ratings).encode('utf-8'))

    def show_page_footer(self):
        # 顯示頁尾
        self.wfile.write('<hr>'.encode('utf-8'))
        self.wfile.write('<p>本系統由 <a href="https://github.com/JoyWannn" target="blank">Joy Wan</a> 製作，'
                         '所有商品相關資料取自Amazon網站，版權歸原作者。</p>'.encode('utf-8'))
        self.wfile.write('</body></html>'.encode('utf-8'))

    # 顯示用戶對於不同商品類型的偏好
    def show_user_genre_preference(self, n_cols=6):
        # 取得用戶對於不同電影類型的偏好，並將結果儲存到 genre_preference 變數中，該變數為一個字典，其中的鍵為電影類型，值為偏好
        genre_preference = rec.user_history.get_user_genre_preference(rec.item_database)

        # 將 genre_preference 變數中的電影類型依照偏好由大到小排序
        genre_preference = sorted(genre_preference.items(), key=lambda x: x[1], reverse=True)

        # 顯示用戶對於不同電影類型的偏好，一行最多顯示 n_cols 個類別
        self.wfile.write('<h2>用戶 {} 共有 {} 個評分紀錄，對於不同商品類型的偏好比例</h2>'.format(
            rec.user_history.user_id, rec.user_history.get_rating_count()).encode('utf-8'))
        self.wfile.write('<table>'.encode('utf-8'))

        # 顯示表格的標題
        self.wfile.write('<tr>'.encode('utf-8'))
        for i in range(0, n_cols):
            self.wfile.write('<th>商品類型</th><th>偏好</th>'.encode('utf-8'))
        self.wfile.write('</tr>'.encode('utf-8'))

        for i in range(0, len(genre_preference), n_cols):  # 每個橫列，最多顯示六個電影類型
            self.wfile.write('<tr>'.encode('utf-8'))
            for j in range(i, i + n_cols):  # 每個直行，從第 i 個電影類型開始
                if j < len(genre_preference):
                    self.wfile.write('<td>{}</td><td>{:.2f}</td>'.format(
                        genre_preference[j][0], genre_preference[j][1]).encode('utf-8'))
            self.wfile.write('</tr>'.encode('utf-8'))
        self.wfile.write('</table>'.encode('utf-8'))

    def show_top_rated_movies(self):
        # 取得用戶最喜歡的物品編號清單，以及這些物品的詳細資訊
        top_rated_product_id_list = rec.user_history.get_top_rated_items(rec.num_recommend)
        top_rated_items = rec.item_database.get_product_by_id_list(top_rated_product_id_list)

        # 顯示用戶對於不同電影類型的偏好
        self.show_user_genre_preference()

        self.wfile.write('<h2>用戶 {} 最喜歡的商品</h2>'.format(rec.user_history.user_id).encode('utf-8'))
        self.wfile.write('<table>'.encode('utf-8'))

        # 如果用戶沒有評分紀錄，則顯示「無」，否則顯示表格的標題
        if len(top_rated_items) > 0:
            self.wfile.write(
                '<tr><th></th><th>評分</th><th width=90>標題</th><th>簡介</th></tr>'.encode('utf-8'))
        else:
            self.wfile.write('<tr><td>無</td></tr>'.encode('utf-8'))

        # 逐筆顯示用戶最喜歡的物品的詳細資訊，包括編號、評分、標題、簡介，以及用戶對這些物品的評分，以便用戶可以修改評分
        for i, product in enumerate(top_rated_items):
            user_rating = rec.user_history.get_user_rating(product)
            self.wfile.write('<tr><td>#{}</td><td>'.format(i).encode('utf-8'))
            self.wfile.write('<select name="{}{}">'.format(self.KEY_RATING_PREFIX, product['asin']).encode('utf-8'))
            self.wfile.write('<option value=""></option>'.encode('utf-8'))
            for j in self.VALID_RATINGS:
                selected = ' selected' if j == user_rating else ''
                self.wfile.write('<option value="{}"{}>{}</option>'.format(j, selected, j).encode('utf-8'))
            self.wfile.write('</select>'.encode('utf-8'))
            self.wfile.write('</td>'.encode('utf-8'))
            #網址要再改
            self.wfile.write('<td><a href="https://www.themoviedb.org/movie/{}" target="_blank">{}</a></td>'.format(
                product['asin'], product['title']).encode('utf-8'))
            self.wfile.write('<td>{}</td>'.format(product['description']).encode('utf-8'))
            self.wfile.write('</tr>'.encode('utf-8'))
        self.wfile.write('</table>'.encode('utf-8'))

    def show_recommendations(self):
        # 顯示推薦清單，包括評分和標題，以表格呈現。其中，評分為輸入下拉式欄位，用戶可以選擇評分，0表示最不喜歡，5表示最喜歡
        self.wfile.write('<h2>用戶 {} 的個人化推薦清單</h2>'.format(rec.user_history.user_id).encode('utf-8'))

        self.wfile.write('<table>'.encode('utf-8'))

        # 如果用戶沒有評分紀錄，則顯示「無」，否則顯示表格的標題
        if len(rec.last_recommendation) > 0:
            self.wfile.write(
                '<tr><th></th><th>評分</th><th width=90>標題</th><th>圖片</th><th>簡介</th></tr>'.encode('utf-8'))
        else:
            self.wfile.write('<tr><td>無</td></tr>'.encode('utf-8'))

        # 逐筆顯示推薦清單的詳細資訊，包括編號、評分、標題、圖片、簡介，以及用戶對這些物品的評分，以便用戶可以修改評分
        for i, product in enumerate(rec.last_recommendation):
            self.wfile.write('<tr><td>#{}</td><td>'.format(i).encode('utf-8'))
            #這邊是評分區
            self.wfile.write('<select name="{}{}">'.format(self.KEY_RATING_PREFIX, product['asin']).encode('utf-8'))

            self.wfile.write('<option value=""></option>'.encode('utf-8'))
            user_rating = rec.user_history.get_user_rating(product)
            for j in self.VALID_RATINGS:
                selected = ' selected' if j == user_rating else ''
                self.wfile.write('<option value="{}"{}>{}</option>'.format(j, selected, j).encode('utf-8'))
            self.wfile.write('</select>'.encode('utf-8'))
            self.wfile.write('</td>'.encode('utf-8'))
            # 網址要再改
            self.wfile.write('<td><a href="https://www.themoviedb.org/movie/{}" target="_blank">{}</a></td>'.format(
                product['asin'], product['title']).encode('utf-8'))
            self.wfile.write('<td>{}</td>'.format(product['overview']).encode('utf-8'))
            self.wfile.write('</tr>'.encode('utf-8'))
        self.wfile.write('</table>'.encode('utf-8'))

    def show_main_page(self):
        # 顯示推薦系統主頁面，包括標題、用戶最喜歡的電影、推薦清單，以及用戶可以給予評分的表單

        # 設定HTTP回應的狀態碼、標頭和內容類型
        self.send_response(200)
        self.send_header('Content-type', 'text/html; charset=utf-8')
        self.end_headers()

        # 顯示標題
        self.show_page_title()

        # 如果未曾計算過推薦清單，則初始化推薦清單
        if rec.last_recommendation is None:
            rec.refresh_recommendation(method='user_based')

        # 建立一個表單，用戶可以看到喜歡的電影與推薦的電影，並在其中選擇部分或全部電影給予評分，並提交表單
        self.wfile.write('<form action="/feedback" method="post">'.encode('utf-8'))

        # 顯示最受用戶歡迎的電影，以及推薦清單
        self.show_top_rated_movies()
        self.show_recommendations()

        # 顯示提交表單的按鈕
        self.wfile.write('<input type="submit" value="更新評分" />'.encode('utf-8'))
        self.wfile.write('<input type="hidden" name="user_id" value="{}" />'.format(
            rec.user_history.user_id).encode('utf-8'))
        self.wfile.write('</form>'.encode('utf-8'))

        # 顯示更新推薦清單的按鈕
        self.wfile.write('<p><a href="refresh"><input type="button" value="更新推薦清單" /></a></p>'.encode('utf-8'))

        # 顯示切換使用者的輸入表單
        self.wfile.write('<form action="/switch_user" method="post">'.encode('utf-8'))
        self.wfile.write('<input type="submit" value="切換使用者為" />'.encode('utf-8'))
        self.wfile.write('<input type="text" name="{}" value="{}" size="6" width="6"/>'.format(
            self.KEY_SWITCH_USER, rec.user_history.user_id).encode('utf-8'))
        self.wfile.write('(若輸入 {} 則可以登入為新用戶)'.format(rec.user_history.get_max_user_id() + 1).encode('utf-8'))
        self.wfile.write('</form>'.encode('utf-8'))

        # 顯示存檔並結束系統的按鈕
        self.wfile.write('<p><a href="quit"><input type="button" value="存檔並結束系統" /></a></p>'.encode('utf-8'))

        # 顯示頁尾
        self.show_page_footer()

    def do_GET(self, web=None):
        print('收到HTTP GET請求，路徑是', self.path)
        # 依據路徑的不同，顯示不同的頁面

        # 如果路徑是根目錄，則顯示主頁面
        if self.path == '/':
            self.show_main_page()

        # 如果路徑是/refresh，則更新物品特徵資料庫，並更新推薦清單
        elif self.path == '/refresh':

            rec.update_item_features()  # 更新物品特徵資料庫
            rec.refresh_recommendation(method='user_based')  # 更新推薦清單
            #rec.refresh_recommendation(method= 'item_based') #待用

            # 設定HTTP回應的狀態碼、標頭和內容類型
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.show_page_title()

            # 顯示更新推薦清單的訊息
            self.wfile.write('<p>已經更新推薦清單</p>'.encode('utf-8'))
            self.wfile.write('<p><a href="/"><input type="button" value="回到首頁" /></a></p>'.encode('utf-8'))

            # 顯示頁尾
            self.show_page_footer()

        # 如果路徑是/quit，則停止伺服器
        elif self.path == '/quit':
            rec.user_history.save_to_disk()  # 將用戶歷史紀錄存檔

            # 設定HTTP回應的狀態碼、標頭和內容類型
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.show_page_title()

            # 顯示結束系統的訊息
            self.wfile.write('<p>系統已經終止，請確認Python程式已經完全關閉，以免無法重啟Web伺服器。</p>'.encode('utf-8'))
            self.wfile.write('<p><a href="/"><input type="button" value="回到首頁" /></a></p>'.encode('utf-8'))

            # 顯示頁尾
            self.show_page_footer()

            # 停止伺服器
            web.stop_server()

        # 其他路徑則顯示404錯誤
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.show_page_title()
            self.wfile.write('<p>你走錯地方了(404)</p>'.encode('utf-8'))
            self.wfile.write('<p><a href="/"><input type="button" value="回到首頁" /></a></p>'.encode('utf-8'))
            self.wfile.write('</body></html>'.encode('utf-8'))

    def do_POST(self):
        print('收到HTTP POST請求，路徑是', self.path)
        # 依據路徑的不同，處理不同的表單，並顯示不同的頁面

        # 如果路徑是/feedback，則處理用戶的評分表單
        if self.path == '/feedback':
            # 讀取表單內容
            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len)
            post_body = post_body.decode('utf-8')
            print('收到的表單內容是', post_body)

            # 解析表單內容，例如：rating_1=5&rating_2=4&rating_3=3，並更新用戶的評分紀錄
            for item in post_body.split('&'):
                key, value = item.split('=')

                # 如果評分是空字串(用戶沒有給分)，則跳過
                if value == '':
                    continue

                # 如果是有數值的評分欄位，則更新用戶的評分紀錄
                if key.startswith(self.KEY_RATING_PREFIX):
                    # 例如：rating_1=4.5，則取出1，並將評分4.5設定給movie_id=1的電影
                    movie_id = int(key[len(self.KEY_RATING_PREFIX):])
                    movie_rated = rec.item_database.get_product_by_id(movie_id)
                    rec.user_history.set_rating(movie_rated, float(value))

            # 將用戶的評分紀錄存檔
            rec.user_history.save_to_disk()

            # 設定HTTP回應的狀態碼、標頭和內容類型
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.show_page_title()

            # 顯示更新用戶的評分紀錄的訊息
            self.wfile.write('<p>已經更新用戶的評分紀錄，可以繼續更新其他評分。若有需要，可自行啟動推薦清單的更新。</p>'.encode('utf-8'))
            self.wfile.write('<p><a href="/"><input type="button" value="回到首頁" /></a></p>'.encode('utf-8'))

            # 顯示頁尾
            self.show_page_footer()

        # 如果是切換用戶
        elif self.path == '/switch_user':
            # 讀取表單內容
            content_len = int(self.headers.get('Content-Length'))
            post_body = self.rfile.read(content_len)
            post_body = post_body.decode('utf-8')
            print('收到的表單內容是', post_body)

            # 解析表單內容，例如：user_id=123，並切換用戶為123
            for item in post_body.split('&'):
                key, value = item.split('=')

                # 如果是用戶ID欄位，則切換用戶
                if key == self.KEY_SWITCH_USER:
                    try:
                        rec.user_history.user_id = int(value)  # 假設用戶ID只有數字
                        rec.last_recommendation = None  # 清除上次的推薦清單，以便下次重新計算
                    except ValueError:
                        print('用戶ID不是數字，無法切換用戶')

            # 設定HTTP回應的狀態碼、標頭和內容類型
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.show_page_title()

            # 顯示切換用戶的訊息
            self.wfile.write('<p>已經切換用戶為{}</p>'.format(rec.user_history.user_id).encode('utf-8'))
            # 提示用戶還有其他用戶帳號可選
            self.wfile.write('<p>你也可以選擇以下範例用戶帳號：<br>'.encode('utf-8'))
            #這邊要修正範例用戶
            self.wfile.write('564是喜劇片愛好者，297是驚悚片愛好者，149是科幻片愛好者，12是愛情片愛好者，571是恐怖片愛好者。</p>'.encode('utf-8'))
            self.wfile.write('<p><a href="/"><input type="button" value="回到首頁" /></a></p>'.encode('utf-8'))

            # 顯示頁尾
            self.show_page_footer()

        # 其他路徑則顯示404錯誤
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.show_page_title()
            self.wfile.write('<p>你走錯地方了(404)</p>'.encode('utf-8'))
            self.wfile.write('<p><a href="/"><input type="button" value="回到首頁" /></a></p>'.encode('utf-8'))
            self.wfile.write('</body></html>'.encode('utf-8'))
class Web:
    # 這個類別是為了啟動Web伺服器，並在伺服器停止時，關閉Python程式，方便在命令列中執行，而不是在IDE中執行
    def __init__(self):
        self.server = None

    def start_server(self):
        # 啟動Web伺服器，並在伺服器停止時，關閉Python程式
        self.server = http.HTTPServer(('localhost', Config.WEB_SERVER_PORT), RequestHandler)
        print('網頁伺服器已經啟動，請開啟瀏覽器，輸入網址 http://localhost:{}/'.format(Config.WEB_SERVER_PORT))
        print('按下Ctrl+C可以停止伺服器')
        self.server.serve_forever()
        print('網頁伺服器已經停止')

    def stop_server(self):
        # 停止Web伺服器
        print('停止網頁伺服器')
        self.server.shutdown()


if __name__ == '__main__':
    # 建立推薦系統，持有所有的資料庫和推薦演算法
    rec = RecommenderSim()

    if not Config.WEB_MODE:
        # 在命令列中執行推薦系統，方便debug
        rec.main_loop()
    else:
        # 啟動Web伺服器，並在伺服器停止時，關閉Python程式
        web = Web()
        web.start_server()

    print('程式結束')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

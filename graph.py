import numpy as np
from numpy import NaN
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt 
import itertools
import networkx as nx

class Grapher:
    """
    Description:
            This class is used to generate:
                    1) the graph (in dictionary form) { source_node: [destination_node1, destination_node2]}
                    2) the dataframe with relative_distances 

    Inputs: The class consists of a pandas dataframe consisting of cordinates for bounding boxe and the image of the invoice/receipt. 

    """
    def __init__(self, file_path):
        self.file_path = file_path
        image_path = file_path.replace(".csv", ".jpg").replace("/csv/", "/images/")
        self.df = pd.read_csv(file_path, sep=",")
        # print(image_path)
        self.image = cv2.imread(image_path)
        
    def graph_formation(self, export_graph = False):

        """
        Description:
        ===========
        Line formation:
        1) Sort words based on Top coordinate:
        2) Form lines as group of words which obeys the following:
            Two words (W_a and W_b) are in same line if:
                Top(W_a) <= Bottom(W_b) and Bottom(W_a) >= Top(W_b)
        3) Sort words in each line based on Left coordinate

        This ensures that words are read from top left corner of the image first, 
        going line by line from left to right and at last the final bottom right word of the page is read.
    
        Args: 
            df with words and cordinates (xmin,xmax,ymin,ymax)
            image read into cv2
        returns: 
            df with words arranged in orientation top to bottom and left to right, the line number for each word, index of the node connected to
            on all directions top, bottom, right and left (if they exist and satisfy the parameters provided)

        _____________________y axis______________________
        |
        |                       top    
        x axis               ___________________
        |              left | bounding box      |  right
        |                   |___________________|           
        |                       bottom 
        |
        |


        iterate through the rows twice to compare them.
        remember that the axes are inverted.
      
        """
        df = self.df
        """
        preprocessing the raw csv files to favorable df 
        """
        #  sort from top to bottom
        df.sort_values(by=['ymin'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        #  subtracting ymax by 1 to eliminate ambiguity of boxes being in both left and right 
        df["ymax"] = df["ymax"].apply(lambda x: x - 1)
        # df["ymin"] = df["ymin"].apply(lambda x: x + 1)

        master = []
        for idx, row in df.iterrows():
            # flatten the nested list 
            flat_master = list(itertools.chain(*master))

            # check to see if idx is in flat_master
            if idx not in flat_master:
                top_a = row['ymin']
                bottom_a = row['ymax']         
                # every line will atleast have the word in it
                line = [idx]         
                for idx_2, row_2 in df.iterrows():
                    # check to see if idx_2 is in flat_master removes ambiguity
                    # picks higher cordinate one. 
                    if idx_2 not in flat_master:
                    # if not the same words
                        if not idx == idx_2:
                            top_b = row_2['ymin']
                            bottom_b = row_2['ymax'] 
                            if (top_a <= bottom_b) and (bottom_a - ((bottom_a-top_a)//2) >= top_b): 
                                line.append(idx_2)
            
                master.append(line)

        df2 = pd.DataFrame({'words_indices': master, 'line_number':[x for x in range(1,len(master)+1)]})

        # explode the list columns eg : [1,2,3]
        df2 = df2.set_index('line_number').words_indices.apply(pd.Series).stack()\
                .reset_index(level=0).rename(columns={0:'words_indices'})
        df2['words_indices'] = df2['words_indices'].astype('int')

        # put the line numbers back to the list
        final = df.merge(df2, left_on=df.index, right_on='words_indices')
        final.drop('words_indices', axis=1, inplace=True)

        """
        3) Sort words in each line based on Left coordinate
        """
        final2 =final.sort_values(by=['line_number','xmin'],ascending=True)\
                .groupby('line_number')\
                .head(len(final))\
                .reset_index(drop=True)

        df = final2 
        
        """
        Pseudocode:
        1) Read words from each line starting from topmost line going towards bottommost line
        2) For each word, perform the following:
            - Check words which are in vertical projection with it.
            - Calculate RD_l and RD_r for each of them 
            - Select nearest neighbour words in horizontal direction which have least magnitude of RD_l and RD_r, 
            provided that those words do not have an edge in that direciton.
                    - In case, two words have same RD_l or RD_r, the word having higher top coordinate is chosen.
            - Repeat steps from 2.1 to 2.3 similarly for retrieving nearest neighbour words in vertical direction by 
            taking horizontal projection, calculating RD_t and RD_b and choosing words having higher left co-ordinate
            incase of ambiguity
            - Draw edges between word and its 4 nearest neighbours if they are available.

        Args: 
            df after lines properly aligned
            
        returns: 
            graph in the form of a dictionary, networkX graph, dataframe with 

        """

        # horizontal edges formation
        df.reset_index(inplace=True)
        grouped = df.groupby('line_number')
        # pd.set_option('display.max_rows', df.shape[0]+1)
        # print(df)
        # exit()
        # for undirected graph construction
        horizontal_connections = {}
        # left
        left_connections = {}    
        # right
        right_connections = {}

        for _,group in grouped:
            a = group['index'].tolist()
            b = group['index'].tolist()
            horizontal_connection = {a[i]:a[i+1] for i in range(len(a)-1) }
            # storing directional connections
            right_dict_temp = {a[i]:{'right':a[i+1]} for i in range(len(a)-1) }
           
            left_dict_temp = {b[i+1]:{'left':b[i]} for i in range(len(b)-1) }
            
            # trong trường hợp không có box nào ở bên phải 
            if len(right_dict_temp) == 0:
                for i in range(len(a)):  
                    df.loc[df['index'] == a[i], 'right'] = NaN
            
            else:
                for i in range(len(a)-1):  
                    df.loc[df['index'] == a[i], 'right'] = int(a[i+1])
            
            # trong trường hợp không có box nào ở bên trái 
            if len(left_dict_temp) == 0:
                for i in range(len(a)):  
                    df.loc[df['index'] == a[i], 'left'] = NaN
            else:
                for i in range(len(a)-1):   
                    df.loc[df['index'] == a[i+1], 'left'] = int(a[i])
        
            left_connections.update(right_dict_temp)
            right_connections.update(left_dict_temp)
            horizontal_connections.update(horizontal_connection)

        dic1, dic2 = left_connections, right_connections
                
        # verticle connections formation
        bottom_connections = {}
        top_connections = {}
        distance_connection = {}

        for src_idx, src_row in df.iterrows():
            
            # ================= vertical ======================= #
            src_range_x = (src_row['xmin'], src_row['xmax'])
            src_center_y = (src_row['ymin'] + src_row['ymax'])/2
            src_center_x = (src_row['xmin'] + src_row['xmax'])/2

            dest_attr_vert = []

            for dest_idx, dest_row in df.iterrows():
                # flag to signal whether the destination object is below source
                is_beneath = False
                if not src_idx == dest_idx:
                    # ==================== vertical ==========================#
                    dest_range_x = (dest_row['xmin'], dest_row['xmax'])
                    dest_center_y = (dest_row['ymin'] + dest_row['ymax'])/2
                    dest_center_x = (dest_row['xmin'] + dest_row['xmax'])/2
                    
                    # height = (dest_center_y - src_center_y)**2 + (dest_center_x - src_center_x)**2
                    height = dest_center_y - src_center_y
                    # if src_row["line_number"] != dest_row["line_number"] - 1:
                    #     continue

                    # consider only the cases where destination object lies 
                    # below source
                    if dest_center_y > src_center_y:
                        # check if horizontal range of dest lies within range 
                        # of source

                        # case 1
                        if dest_range_x[0] <= src_range_x[0] and \
                            dest_range_x[1] >= src_range_x[1]:
                            
                            x_common = (src_range_x[0] + src_range_x[1])/2
                            
                            line_src = (x_common , src_center_y)
                            line_dest = (x_common, dest_center_y)

                            attributes = (dest_idx, line_src, line_dest, height,src_idx)
                            dest_attr_vert.append(attributes)
                            
                            is_beneath = True

                        # case 2
                        elif dest_range_x[0] >= src_range_x[0] and \
                            dest_range_x[1] <= src_range_x[1]:
                            
                            x_common = (dest_range_x[0] + dest_range_x[1])/2
                            
                            line_src = (x_common, src_center_y)
                            line_dest = (x_common, dest_center_y)
                            
                            attributes = (dest_idx, line_src, line_dest, height,src_idx)
                            dest_attr_vert.append(attributes)
                            
                            is_beneath = True

                        # case 3
                        elif dest_range_x[0] <= src_range_x[0] and \
                            dest_range_x[1] >= src_range_x[0] and \
                                dest_range_x[1] < src_range_x[1]:

                            x_common = (src_range_x[0] + dest_range_x[1])/2

                            line_src = (x_common , src_center_y)
                            line_dest = (x_common, dest_center_y)

                            attributes = (dest_idx, line_src, line_dest, height,src_idx)
                            dest_attr_vert.append(attributes)

                            is_beneath = True

                        # case 4
                        elif dest_range_x[0] <= src_range_x[1] and \
                            dest_range_x[1] >= src_range_x[1] and \
                                dest_range_x[0] > src_range_x[0]:
                            
                            x_common = (dest_range_x[0] + src_range_x[1])/2
                            
                            line_src = (x_common , src_center_y)
                            line_dest = (x_common, dest_center_y)

                            attributes = (dest_idx, line_src, line_dest, height,src_idx)
                            dest_attr_vert.append(attributes)

                            is_beneath = True
            
            dest_attr_vert_sorted = sorted(dest_attr_vert, key = lambda x: x[3])
            if len(dest_attr_vert_sorted) == 0:
                pass
            else:
                # if src_idx in bottom_connections or dest_idx in top_connections:
                #     continue
                src_idx = dest_attr_vert_sorted[0][4]
                dest_idx = dest_attr_vert_sorted[0][0]
                distance = dest_attr_vert_sorted[0][3]
                if dest_idx not in top_connections.keys():
                    bottom_connections[src_idx] = dest_idx                
                    top_connections[dest_idx] = src_idx
                    distance_connection[dest_idx] = distance

                    df.loc[df['index'] == src_idx , 'bottom'] = dest_idx
                    df.loc[df['index'] == dest_idx, 'top'] = src_idx
                else:
                    if (dest_idx in distance_connection.keys() and distance_connection[dest_idx] > distance):
                        
                        del bottom_connections[top_connections[dest_idx]]
                        df.loc[df['index'] == top_connections[dest_idx], 'bottom'] = np.NaN

                        bottom_connections[src_idx] = dest_idx                
                        top_connections[dest_idx] = src_idx
                        distance_connection[dest_idx] = distance

                        df.loc[df['index'] == src_idx , 'bottom'] = dest_idx
                        df.loc[df['index'] == dest_idx, 'top'] = src_idx

        # print(df)  
        # combining both 
        result = {}
        dic1 = horizontal_connections
        dic2 = bottom_connections

        for key in (dic1.keys() | dic2.keys()):
            if key in dic1: result.setdefault(key, []).append(dic1[key])
            if key in dic2: result.setdefault(key, []).append(dic2[key])

        G = nx.from_dict_of_lists(result)
        
        # draw graph
        if export_graph:

            if not os.path.exists('./GCN_data/draw_images'):
                os.makedirs('./GCN_data/draw_images')			

            file_name = os.path.basename(self.file_path).split(".")[0]
            plot_path = f'./GCN_data/draw_images/{file_name }_graph.png'
            print(plot_path)
            layout = nx.kamada_kawai_layout(G)   
            layout = nx.spring_layout(G)     
            nx.draw(G, layout, with_labels=True)
            plt.savefig(plot_path, format="PNG", dpi=600)

        self.df = df 
        return G, result, df 

    # features calculation    
    def get_text_features(self, df): 
        """
        gets text features 

        Args: df
        Returns: n_upper, n_alpha, n_numeric
        """
        data = df['object'].tolist()
        
        '''
            Args:
                df
                
            Returns: 
                character and word features
                
        '''

        #  character wise
        n_upper, n_alpha, n_numeric = [], [], []

        for words in data:
            lower, upper, alpha, numeric = 0,0,0,0
            for char in words: 
    
                #  for upper letters 
                if char.isupper(): 
                    upper += 1 

                #  for alphabetic chars
                if char.isalpha():
                    alpha += 1  

                #  for numeric chars
                if char.isnumeric():
                    numeric += 1                            

            n_upper.append(upper)
            n_alpha.append(alpha)
            n_numeric.append(numeric)

        df['n_upper'], df['n_alpha'],\
        df['n_numeric'] = n_upper, n_alpha, n_numeric

    def relative_distance(self, export_document_graph = False):
        """ 
        1) Calculates relative distances for each node in left, right, top  and bottom directions if they exist.
        rd_l, rd_r = relative distances left , relative distances right. The distances are divided by image width
        rd_t, rd_b = relative distances top , relative distances bottom. The distances are divided by image length

        2) Exports the complete document graph for visualization

        Args: 
            result dataframe from graph_formation()
             
        returns: 
            dataframe with features and exports document graph if prompted
        """


        df, img = self.df, self.image
        image_height, image_width = self.image.shape[0], self.image.shape[1]

        for index in df['index'].to_list():
            right_index = df.loc[df['index'] == index, 'right'].values[0]
            left_index = df.loc[df['index'] == index, 'left'].values[0]
            bottom_index = df.loc[df['index'] == index, 'bottom'].values[0]
            top_index = df.loc[df['index'] == index, 'top'].values[0]

            # check if it is nan value 
            if np.isnan(right_index) == False: 
                right_word_left = df.loc[df['index'] == right_index, 'xmin'].values[0]
                source_word_right = df.loc[df['index'] == index, 'xmax'].values[0]
                df.loc[df['index'] == index, 'rd_r'] = (right_word_left - source_word_right)/image_width

                """
                for plotting purposes
                getting the mid point of the values to draw the lines for the graph
                mid points of source and destination for the bounding boxes
                """
                right_word_x_max = df.loc[df['index'] == right_index, 'xmax'].values[0]
                right_word_y_max = df.loc[df['index'] == right_index, 'ymax'].values[0]
                right_word_y_min = df.loc[df['index'] == right_index, 'ymin'].values[0]

                df.loc[df['index'] == index, 'destination_x_hori'] = (right_word_x_max + right_word_left)/2
                df.loc[df['index'] == index, 'destination_y_hori'] = (right_word_y_max + right_word_y_min)/2
            else:
                df.loc[df['index'] == index, 'rd_r'] = NaN

            if np.isnan(left_index) == False:
                left_word_right = df.loc[df['index'] == left_index, 'xmax'].values[0]
                source_word_left = df.loc[df['index'] == index, 'xmin'].values[0]
                df.loc[df['index'] == index, 'rd_l'] = (left_word_right - source_word_left)/image_width
            else:
                df.loc[df['index'] == index, 'rd_l'] = NaN
            
            if np.isnan(bottom_index) == False:
                bottom_word_top = df.loc[df['index'] == bottom_index, 'ymin'].values[0]
                source_word_bottom = df.loc[df['index'] == index, 'ymax'].values[0]
                df.loc[df['index'] == index, 'rd_b'] = (bottom_word_top - source_word_bottom)/image_height

                """for plotting purposes"""
                bottom_word_top_max = df.loc[df['index'] == bottom_index, 'ymax'].values[0]
                bottom_word_x_max = df.loc[df['index'] == bottom_index, 'xmax'].values[0]
                bottom_word_x_min = df.loc[df['index'] == bottom_index, 'xmin'].values[0]
                df.loc[df['index'] == index, 'destination_y_vert'] = (bottom_word_top_max + bottom_word_top)/2
                df.loc[df['index'] == index, 'destination_x_vert'] = (bottom_word_x_max + bottom_word_x_min)/2

            else:
                df.loc[df['index'] == index, 'rd_b'] = NaN

            if np.isnan(top_index) == False:
                top_word_bottom = df.loc[df['index'] == top_index, 'ymax'].values[0]
                source_word_top = df.loc[df['index'] == index, 'ymin'].values[0]
                df.loc[df['index'] == index, 'rd_t'] = (top_word_bottom - source_word_top)/image_height
            else:
                df.loc[df['index'] == index, 'rd_t'] = NaN
            


        # replace all tne NaN values with '0' meaning there is nothing in that direction
        df[['rd_r','rd_b','rd_l','rd_t']] = df[['rd_r','rd_b','rd_l','rd_t']].fillna(0)

        if export_document_graph:
            for idx, row in df.iterrows():
                cv2.rectangle(img, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 0, 255), 2)

                if 'destination_x_vert' in row:
                    if np.isnan(row['destination_x_vert']) == False:
                        source_x = (row['xmax'] + row['xmin'])/2
                        source_y = (row['ymax'] + row['ymin'])/2
                        
                        cv2.line(img, 
                                (int(source_x), int(source_y)),
                                (int(row['destination_x_vert']), int(row['destination_y_vert'])), 
                                (0,255,0), 2)
                        cv2.circle(img, (int(source_x), int(source_y)), 8, (255,0,0), -1)
                        cv2.circle(img, (int(row['destination_x_vert']), int(row['destination_y_vert'])), 8, (255,0,0), -1)


                        text = "{:.3f}".format(row['rd_b'])
                        text_coordinates = ( int((row['destination_x_vert'] + source_x)/2) , int((row['destination_y_vert'] +source_y)/2))     
                        cv2.putText(img, text, text_coordinates, cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,0,0), 1)

                    # text_coordinates = ((row['destination_x_vert'] + source_x)/2 , (row['destination_y_vert'] +source_y)/2)
                
                if 'destination_x_hori' in row:
                    if np.isnan(row['destination_x_hori']) == False:
                        print("row['destination_x_hori']: ", row['destination_x_hori'])
                        source_x = (row['xmax'] + row['xmin'])/2
                        source_y = (row['ymax'] + row['ymin'])/2

                        cv2.line(img, 
                            (int(source_x), int(source_y)),
                            (int(row['destination_x_hori']), int(row['destination_y_hori'])), \
                            (255,0,0), 2)

                        text = "{:.3f}".format(row['rd_r'])
                        text_coordinates = (int((row['destination_x_hori'] + source_x)/2) , int((row['destination_y_hori'] +source_y)/2))     
                        cv2.putText(img, text, text_coordinates, cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,0,0), 1)

                        cv2.circle(img, (int(source_x), int(source_y)), 8, (255,0,0), -1)
                        cv2.circle(img, (int(row['destination_x_hori']), int(row['destination_y_hori'])), 8, (255,0,0), -1)

                if not os.path.exists('./GCN_data/draw_images'):
                    os.makedirs('./GCN_data/draw_images')			
                
                file_name = os.path.basename(self.file_path).split(".")[0]
                plot_path = f'./GCN_data/draw_images/{file_name}_docu_graph.jpg'
                cv2.imwrite(plot_path, img)
   
        # drop the unnecessary columns
        try:
            df.drop(['destination_x_hori', 'destination_y_hori','destination_y_vert','destination_x_vert'], axis=1, inplace=True)
        except:
            pass
        self.get_text_features(df)
        
        return df

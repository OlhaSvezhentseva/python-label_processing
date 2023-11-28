# Import third-party libraries
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

# Suppress warning messages during execution
import warnings
warnings.filterwarnings('ignore')

def calculate_iou(pred_coords: tuple[float, float, float, float], 
                  gt_coords: tuple[str, float, float, float, float]) -> float:
    """
    Calculates IOU scores by comparing the ground truth and predicted 
    segmentation coordinates.

    Args:
        df_pred (pd.Dataframe): path to predicted coordinates csv
        df_gt (pd.Dataframe): path to ground truth coordinates csv
    
    Returns:
        iou (float): iou scores for each prediction
    """
    #Extract bounding boxes coordinates
    xmin_pred, ymin_pred, xmax_pred, ymax_pred = pred_coords
    _, xmin_gt, ymin_gt, xmax_gt, ymax_gt = gt_coords
    # Get the coordinates of the intersection rectangle
    x0_I = max(xmin_pred, xmin_gt)
    y0_I = max(ymin_pred, ymin_gt)
    x1_I = min(xmax_pred, xmax_gt)
    y1_I = min(ymax_pred, ymax_gt) 
    #Calculate width and height of the intersection area
    width_I = x1_I - x0_I 
    height_I = y1_I - y0_I  
    #Handle the negative value width or height of the intersection area
    if width_I<0 : width_I=0
    if height_I<0 : height_I=0
    # Calculate the intersection area:
    intersection = width_I * height_I   
    # Calculate the union area:
    width_A, height_A = xmax_pred - xmin_pred, ymax_pred - ymin_pred
    width_B, height_B = xmax_gt - xmin_gt, ymax_gt - ymin_gt
    union = (width_A * height_A) + (width_B * height_B) - intersection   
    #Calculate the IOU:
    iou: float = intersection/union      
    #Return the IOU and intersection box
    return iou

def comparison(df_pred_filename: pd.DataFrame,
               df_gt_filename: pd.DataFrame) -> pd.DataFrame:
    """
    For one unique jpg filename this function uses the bounding box-coordinates 
    of each predicted label and calculates for every label of the ground truth 
    the iou-score. Then it takes the maximum score and adds it to the dataframe. 

    Args:
        df_pred_filename (pd.DataFrame): subdataframe of predicted labels containing all the rows belonging to one filename
        df_gt_filename (pd.DataFrame):  subdataframe of groundtruth containing all the rows belonging to one filename

    Returns:
        pd.DataFrame: new sub-dataframe with coordinates of ground truth and 
            predicted labels as well as the (max) iou score
    """
    max_scores = []
    max_coords = []
    for _, row_pred in df_pred_filename.iterrows():
        max_score: float = 0
        pred_coords = (row_pred.xmin_pred, row_pred.ymin_pred,
                       row_pred.xmax_pred, row_pred.ymax_pred)
        for _, row_gt in df_gt_filename.iterrows():
            gt_coords = (row_gt.class_gt, row_gt.xmin_gt,
                         row_gt.ymin_gt, row_gt.xmax_gt, row_gt.ymax_gt)
            iou = calculate_iou(pred_coords, gt_coords)
            if iou > max_score:
                max_score = iou
                max_coord = gt_coords
        max_coords.append(max_coord)
        max_scores.append(max_score)
    df_pred_filename["score"] = max_scores
    df_pred_filename[["class_gt","xmin_gt", "ymin_gt",
                      "xmax_gt", "ymax_gt"]] = max_coords
    return df_pred_filename


def concat_frames(df_pred: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    """
    Concats predicted and groundtruth dataset with the coordinates' IOU scores.

    Args:
        df_pred (pd.DataFrame): dataframe with predicted bounding boxes from segmentation
        df_gt (pd.DataFrame): dataframe containing the groundtruth 

    Returns:
        pd.DataFrame: Concatenated dataframe with IOU scores.
    """
    
    df_gt.rename(columns={"class": "class_gt", "xmin": "xmin_gt",
                          "ymin": "ymin_gt", "xmax": "xmax_gt",
                          "ymax": "ymax_gt"},
                 inplace=True)
    df_pred.rename(columns={"class": "class_pred", "xmin": "xmin_pred",
                            "ymin": "ymin_pred", "xmax": "xmax_pred",
                            "ymax": "ymax_pred"},
                   inplace=True)
    frames: list[pd.DataFrame] = []
    for element in df_pred.filename.unique():
        df_pred_filename = df_pred[df_pred.filename == element]
        df_gt_filename = df_gt[df_gt.filename == element]
        new_frame = comparison(df_pred_filename, df_gt_filename)
        frames.append(new_frame)
    df = pd.concat(frames)
    #filepath = Path(f'{folder}/iou_scores.csv')
    #df.to_csv(filepath)
    return df


def box_plot_iou(df_concat: pd.DataFrame) -> go.Figure():
    """
    Creates box plot of the calculated IOU scores for each class.

    Args:
        df_pred (pd.DataFrame): dataframe with predicted bounding boxes from segmentation
        df_gt (pd.DataFrame): dataframe containing the groundtruth 

    Returns:
        pio.Figure: plotly.io graph object
    """
    fig = px.box(df_concat, y="score", points="all", color="class_pred")
    fig.update_layout(title_text="IOU Scores")
    return fig


def class_pred(df_concat: pd.DataFrame) -> go.Figure():
    """
    Creates a bar chart of the predicted and groundtruth classes. 
    Shows the accuracy of the predicted classes of each label compared to the groundtruth.

    Args:
        df_concat (pd.DataFrame): concatenated dataframe
        
    Returns:
        pio.Figure: plotly.io graph object
    """
    iou_df = df_concat
    conditions = [(iou_df['class_pred'] == iou_df['class_gt']), #define conditions
                 (iou_df['class_pred'] != iou_df['class_gt'])]
    choices = ['match', 'no_match'] #define choices
    #create new column in DataFrame that displays results of comparisons
    iou_df['comparison_values'] = np.select(conditions, choices, default='Tie') 
    # Plot
    fig = px.histogram(iou_df, x='class_pred',
                       color="comparison_values").update_xaxes(
                           categoryorder='total descending')
    fig.update_layout(xaxis_type='category', title_text='match VS no_match')
    return fig
    
    

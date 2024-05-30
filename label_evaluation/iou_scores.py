# Import third-party libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from pathlib import Path


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
    For one unique jpg filename, this function uses the bounding box-coordinates 
    of each predicted label and calculates for every label of the ground truth 
    the iou-score. Then it takes the maximum score and adds it to the dataframe. 

    Args:
        df_pred_filename (pd.DataFrame): subdataframe of predicted labels 
            containing all the rows belonging to one filename
        df_gt_filename (pd.DataFrame): subdataframe of ground truth 
            containing all the rows belonging to one filename

    Returns:
        pd.DataFrame: new sub-dataframe with coordinates of ground truth and 
            predicted labels as well as the (max) iou score
    """
    max_scores = []
    max_coords = {'class_gt': [], 'xmin_gt': [], 'ymin_gt': [], 'xmax_gt': [], 'ymax_gt': []}

    for _, row_pred in df_pred_filename.iterrows():
        max_score = 0
        pred_coords = (row_pred.xmin_pred, row_pred.ymin_pred, row_pred.xmax_pred, row_pred.ymax_pred)
        max_coord = {'class_gt': None, 'xmin_gt': None, 'ymin_gt': None, 'xmax_gt': None, 'ymax_gt': None}

        for _, row_gt in df_gt_filename.iterrows():
            gt_coords = (row_gt.class_gt, row_gt.xmin_gt, row_gt.ymin_gt, row_gt.xmax_gt, row_gt.ymax_gt)
            iou = calculate_iou(pred_coords, gt_coords)

            if iou > max_score:
                max_score = iou
                max_coord = {'class_gt': row_gt.class_gt, 'xmin_gt': row_gt.xmin_gt,
                             'ymin_gt': row_gt.ymin_gt, 'xmax_gt': row_gt.xmax_gt, 'ymax_gt': row_gt.ymax_gt}

        max_scores.append(max_score)
        for key, value in max_coord.items():
            max_coords[key].append(value)

    df_pred_filename["score"] = max_scores
    for key, values in max_coords.items():
        df_pred_filename[key] = values

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


def box_plot_iou(df_concat: pd.DataFrame, accuracy_txt_path: str = None) -> go.Figure():
    """
    Creates box plot of the calculated IOU scores for each class.

    Args:
        df_concat (pd.DataFrame): dataframe with predicted and groundtruth bounding boxes
        accuracy_txt_path (str): file path to save accuracy percentages in a text file

    Returns:
        pio.Figure: plotly.io graph object
    """
    # Add a new column indicating whether IOU score is below or above 0.8
    df_concat['category'] = df_concat['score'].apply(lambda x: 'Labels Below 0.8' if x < 0.8 else 'Labels Above 0.8')

    fig = px.box(df_concat, y="score", points="all", color="category", title="IOU Scores Distribution",
                 labels={"score": "IOU Score", "category": "IOU Score Category"})
    fig.update_layout(
        yaxis=dict(title="IOU Scores"),
        legend_title_text="IOU Score Threshold",
        legend=dict(
            traceorder="normal",
            title_font=dict(size=14),
            itemsizing='constant',
            itemclick="toggleothers",
            itemdoubleclick="toggle",
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)"
        ),
    )

    # Calculate the percentage of labels below and above 0.8
    below_08_percentage = (df_concat['category'] == 'Labels Below 0.8').mean() * 100
    above_08_percentage = (df_concat['category'] == 'Labels Above 0.8').mean() * 100

    # Specify custom legend items with percentages
    custom_legend_items = [
        dict(label=f"Box Plot (Labels Below 0.8) - {below_08_percentage:.2f}%", symbol="circle", marker=dict(color="rgba(31, 119, 180, 0.7)")),
        dict(label=f"Box Plot (Labels Above 0.8) - {above_08_percentage:.2f}%", symbol="circle", marker=dict(color="rgba(255, 127, 14, 0.7)")),
    ]

    # Add custom legend items to the legend
    for item in custom_legend_items:
        fig.add_trace(go.Scatter(visible=False, mode="markers", showlegend=True, legendgroup="legend",
                                 name=item["label"], marker=dict(symbol="circle", color=item["marker"]["color"])))

    # Center the title
    fig.update_layout(title=dict(text="Label Detection IOU Scores Distribution", x=0.5))

    # Calculate accuracy percentages and save to text file
    accuracy_df = df_concat.groupby("category")["score"].mean().reset_index()
    accuracy_df["accuracy_percentage"] = accuracy_df["score"] * 100

    if accuracy_txt_path:
        accuracy_txt_path = Path(accuracy_txt_path)
        accuracy_df.to_csv(accuracy_txt_path, index=False)

    return fig
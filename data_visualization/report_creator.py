import os
import subprocess
from jinja2 import Template

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


def report_creator(folder, patient_id):
    player_info_path = os.path.join(folder, r"player_info.csv")
    player_info = pd.read_csv(player_info_path)

    decisions_record_path = os.path.join(folder, r"decisions_record.csv")
    decisions_record = pd.read_csv(decisions_record_path)

    connection_weights_path = os.path.join(folder, r"connection_weights.csv")
    connection_weights = pd.read_csv(connection_weights_path)

    model_evaluation_path = os.path.join(folder, r"model_evaluation.csv")
    model_evaluation = pd.read_csv(model_evaluation_path)

    plot_1_path = generate_plot_1(folder, decisions_record)
    plot_2_path = generate_plot_2(folder, decisions_record)
    plot_3_path = generate_plot_3(folder, decisions_record)
    plot_4_path = generate_plot_4(folder, connection_weights)

    igt_stats = generate_igt_stats(decisions_record)
    comp_stats = generate_comp_stats(connection_weights)

    build_report(folder, patient_id, plot_1_path, plot_2_path, plot_3_path, plot_4_path, player_info.iloc[0].to_dict(),
                 igt_stats, comp_stats, model_evaluation["0"].to_dict())

    # generate pic 1                - decisions_record
    # generate pic 2                - decisions_record
    # generate pic 3                - decisions_record
    # generate pic 4                - connection_weights
    # IGT stats                     - decisions_record
    # Computational Modeling stats  - model_evaluation, connection_weights

    # build report                  - player_info, igt_stats, pic[1-4], comp_stats


def generate_plot_1(folder, decisions_record):
    # Data for the pie chart
    labels = ['A', 'B', 'C', 'D']
    sizes = [decisions_record["Cards chosen from deck (A)"].values[-1],
             decisions_record["Cards chosen from deck (B)"].values[-1],
             decisions_record["Cards chosen from deck (C)"].values[-1],
             decisions_record["Cards chosen from deck (D)"].values[-1]]
    colors = [(1, 0, 0, 0.7), (0, 0, 1, 0.7), (0, 0.5, 0, 0.7), (1, 0.65, 0, 0.7)]

    # Create the pie chart
    font = {'family': 'Garamond', 'size': 8}
    plt.rc('font', **font)

    plt.figure(figsize=(4.9, 2))

    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)

    # Display the pie chart
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

    plt.annotate("Percentage \n of decks", xy=(0, 0.5), xytext=(-0.15, 0.5),
                 textcoords='axes fraction', fontsize=24, fontweight='bold', va='center')

    plot_path = os.path.join(folder, r"plot_1.jpg")

    plt.savefig(plot_path, format='jpeg', dpi=400)

    return plot_path.replace('\\', '/')


def generate_plot_2(folder, decisions_record):
    plt.figure(figsize=(5, 2))

    plt.plot(decisions_record['Cards chosen from deck (A)'], label="A", color='red')
    plt.plot(decisions_record['Cards chosen from deck (B)'], label="B", color='blue')
    plt.plot(decisions_record['Cards chosen from deck (C)'], label="C", color='green')
    plt.plot(decisions_record['Cards chosen from deck (D)'], label="D", color='orange')

    # Add a title and labels
    plt.xlabel("Cards", fontname='Garamond', fontsize=14)
    plt.ylabel("Decisions", fontname='Garamond', fontsize=14)

    # Change the font and size for the legend
    font = {'family': 'Garamond', 'size': 12}

    # Change the font and size for the tick labels
    plt.xticks(fontname='Garamond', fontsize=12)
    plt.yticks(fontname='Garamond', fontsize=12)

    plt.title("Progress of decks", loc="left", fontsize=24, fontweight='bold')

    # Save the figure as JPEG and PNG
    plot_path = os.path.join(folder, r"plot_2.jpg")
    plt.savefig(plot_path, format='jpeg', dpi=400)

    return plot_path.replace('\\', '/')


def generate_plot_3(folder, decisions_record):
    plt.figure(figsize=(5, 4.3))
    plt.plot(decisions_record['Money accumulated'], color='red')
    plt.title("Progress of money", loc="left", fontsize=24, fontweight='bold')

    plot_path = os.path.join(folder, r"plot_3.jpg")
    plt.savefig(plot_path, format='jpeg', dpi=400)

    return plot_path.replace('\\', '/')


def generate_plot_4(folder, connection_weights):
    a = np.array(connection_weights.values)
    plt.figure(figsize=(11.5, 6))

    font = {'family': 'Garamond', 'size': 10}
    plt.rc('font', **font)

    w_xd = np.reshape(a[:76], (19, 4))
    w_xo = np.reshape(a[76:152], (19, 4))

    b = np.concatenate([w_xd, w_xo], axis=1)

    plt.imshow(b.T, cmap='gray', vmin=-1, vmax=1)
    plt.colorbar(label='Weight')

    x_pos = np.arange(19)
    plt.xticks(x_pos, labels=["remaining_decisions", "money", "last_positive", "last_negative", "money_gt_loan",
                              "money_lt_loan", "current_decision", "cards_chosen (Deck)", "last_positive (Deck)",
                              "last_negative (Deck)", "num_positive (Deck)", "num_negative (Deck)",
                              "freq_positive (Deck)", "freq_negative (Deck)", "best_outcome (Deck)",
                              "worst_outcome (Deck)", "unchosen_decisions (Deck)", "won_money (Deck)",
                              "lost_money (Deck)"], rotation=90)

    y_pos = np.arange(8)
    plt.yticks(y_pos,
               labels=["riskL (d)", "riskG (d)", "estimatedL(d)", "estimatedG (d)", "riskL", "riskG", "estimatedL",
                       "estimatedG"])

    plt.tick_params(bottom=False)

    plot_path = os.path.join(folder, r"plot_4.jpg")
    plt.savefig(plot_path, format='jpeg', dpi=400)

    return plot_path.replace('\\', '/')


def generate_igt_stats(decisions_record):
    # Most/least selected deck
    decided_decks = decisions_record['Selected deck'].value_counts()
    most_selected_deck = decided_decks.idxmax()
    least_selected_deck = decided_decks.idxmin()

    # Earned money
    earned_money = decisions_record["Money accumulated"].values[-1]

    # Time mean/range/std
    time_mean = decisions_record["Time"].mean()
    time_range = decisions_record["Time"].max() - decisions_record["Time"].min()
    time_std = decisions_record["Time"].std()

    # Output igt_stats
    igt_stats = {"most_selected_decks": most_selected_deck,
                 "least_selected_decks": least_selected_deck,
                 "earned_money": earned_money,
                 "time_mean": round(time_mean, 2),
                 "time_range": round(time_range, 2),
                 "time_std": round(time_std, 2)
                 }
    return igt_stats


def generate_comp_stats(connection_weights):
    concepts = ["remaining_decisions", "money", "last_positive", "last_negative", "money_gt_loan",
                "money_lt_loan", "current_decision", "cards_chosen (Deck)", "last_positive (Deck)",
                "last_negative (Deck)", "num_positive (Deck)", "num_negative (Deck)",
                "freq_positive (Deck)", "freq_negative (Deck)", "best_outcome (Deck)",
                "worst_outcome (Deck)", "unchosen_decisions (Deck)", "won_money (Deck)",
                "lost_money (Deck)"]

    # reshape weights
    w_xd = np.reshape(connection_weights.values[:76], (19, 4))
    w_xo = np.reshape(connection_weights.values[76:152], (19, 4))

    # Most/least relevant concept
    mag_deliberative = np.linalg.norm(w_xd, axis=1)
    mag_direct = np.linalg.norm(w_xo, axis=1)

    row_magnitudes = np.concatenate([mag_deliberative, mag_direct])
    most_relevant_concept = concepts[np.argmax(row_magnitudes) % 19]
    row_magnitudes = np.where(row_magnitudes == 0, np.inf, row_magnitudes)
    least_relevant_concept = concepts[np.argmin(row_magnitudes) % 19]

    # Mean/std through deliberative and direct
    mean_deliberative = mag_deliberative.mean()
    std_deliberative = mag_deliberative.std()
    mean_direct = mag_direct.mean()
    std_direct = mag_direct.std()

    # T-statistic
    diff_tstat, diff_pvalue = stats.ttest_ind_from_stats(mean_deliberative, std_deliberative, 19, mean_direct,
                                                         std_direct, 19)

    comp_stats = {"most_relevant_concept": most_relevant_concept.replace('_', '\_'),
                  "least_relevant_concept": least_relevant_concept.replace('_', '\_'),
                  "mean_deliberative": round(mean_deliberative, 4),
                  "std_deliberative": round(std_deliberative, 4),
                  "mean_direct": round(mean_direct, 4),
                  "std_direct": round(std_direct, 4),
                  "diff_tstat": round(diff_tstat, 4),
                  "diff_pvalue": round(diff_pvalue, 4)}
    return comp_stats


def build_report(folder, patient_id, plot_1_path, plot_2_path, plot_3_path, plot_4_path, player_info, igt_stats,
                 comp_stats,
                 model_evaluation):
    print(plot_1_path)

    picture_path = r"C:\Users\maial\PycharmProjects\TFG_deliverable\storage_system\patient_93563858\plot_1.jpg"

    picture_path = picture_path.replace('\\', '/')

    # Load the LaTeX template
    with open(r'C:\Users\maial\PycharmProjects\TFG_deliverable\data_visualization\report_template.tex',
              'r') as template_file:
        template = Template(template_file.read())

    # Render the template with the image path
    latex_content = template.render(plot_1_path=plot_1_path, plot_2_path=plot_2_path, plot_3_path=plot_3_path,
                                    plot_4_path=plot_4_path, patient_id=patient_id,
                                    start_side=player_info["Affected Side"],
                                    pathology=player_info["Pathology"], start_date=player_info["Start Date"],
                                    sex=player_info["Sex"],
                                    age=player_info["Age"], most_selected_deck=igt_stats["most_selected_decks"],
                                    least_selected_deck=igt_stats["least_selected_decks"],
                                    money=igt_stats["earned_money"], time_mean=igt_stats["time_mean"],
                                    time_range=igt_stats["time_range"], time_std=igt_stats["time_std"],
                                    epochs=model_evaluation[0],
                                    pop_size=model_evaluation[1], accuracy=model_evaluation[3],
                                    time_corr=model_evaluation[4], train_time=model_evaluation[5],
                                    most_relevant_concept=comp_stats["most_relevant_concept"],
                                    least_relevant_concept=comp_stats["least_relevant_concept"],
                                    mean_deliberative=comp_stats["mean_deliberative"],
                                    std_deliberative=comp_stats["std_deliberative"],
                                    mean_direct=comp_stats["mean_direct"],
                                    std_direct=comp_stats["std_direct"],
                                    diff_tstat=comp_stats["diff_tstat"],
                                    diff_pvalue=comp_stats["diff_pvalue"])

    # Write the LaTeX content to a .tex file
    with open('report.tex', 'w') as output_file:
        output_file.write(latex_content)

    # Compile the LaTeX document to a PDF
    subprocess.run(['pdflatex', '-output-directory=' + folder, 'report.tex'])

    aux_path = os.path.join(folder, r"report.aux")
    log_path = os.path.join(folder, r"report.log")

    subprocess.run(['del', aux_path, log_path], shell=True)
    subprocess.run(['del', plot_1_path.replace('/', '\\'), plot_2_path.replace('/', '\\')], shell=True)
    subprocess.run(['del', plot_3_path.replace('/', '\\'), plot_4_path.replace('/', '\\')], shell=True)
    subprocess.run(['del', 'report.tex'], shell=True)

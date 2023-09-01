import os

from data_visualization.report_creator import report_creator

storage_dir = os.path.join(os.getcwd(), r"storage_system")


class Data_Visualization:
    def __init__(self):
        self.folder = None

    def create_folder(self, patient_id):
        patient_dir = os.path.join(storage_dir, "patient_" + str(patient_id))
        os.mkdir(patient_dir)
        self.folder = patient_dir
        return patient_dir

    def df_to_csv(self, dataframe, filename):
        file_path = os.path.join(self.folder, filename)
        dataframe.to_csv(file_path, index=False)
        return file_path

    def create_report(self, patient_id):
        report_creator(self.folder, patient_id)

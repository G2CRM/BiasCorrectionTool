"""
Bias Correction Tool (GUI)
==========================

A PyQt-based desktop application designed to apply absolute and optional relative
bias adjustments to coastal storm hydrograph simulations (ADCIRC and STWAVE), with
optional hourly resampling centered around peak surge.

Prerequisites & Environment Setup
---------------------------------
Ensure the required Python packages are installed:

    pip install qtpy PyQt5 h5py numpy scipy

    Versions:
        qtpy>=2.4.3
        PyQt6>=6.11.0
        h5py>=3.16.0
        numpy
        scipy>=1.15.3

How to Run
----------
Launch the graphical interface from the command line:

    python bias_correction.py

Workflow & Usage Steps
----------------------
1. Select ADCIRC Input File:
   - Click "Select ADCIRC Input File" and choose the base ADCIRC HDF5 (.h5) file.
2. Select STWAVE Input File:
   - Click "Select STWAVE Input File" and choose the corresponding STWAVE HDF5 (.h5) file.
3. Select ADCIRC Peaks File:
   - Click "Select ADCIRC Peaks File" and choose the peak water elevation HDF5 (.h5) file.
4. Set Output File:
   - Click "Set Output File" and specify destination name and path (*.h5).
5. Configure Parameters:
   - Absolute Bias: Enter the absolute bias value (default: 1.0).
   - Relative Bias (Optional): Enter relative bias value (e.g., 0.5) or leave blank.
   - Resampling: Check "Resample time series at 1-hour increments?" if peak-aligned
     1-hour spacing is needed.
6. Process Files:
   - Click "Process Files" once all input/output paths are configured.
"""

import math
import sys
from datetime import datetime, timedelta

import h5py as h5
import numpy as np
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from scipy.interpolate import interp1d


class MultiInputFileApp(QWidget):
    def __init__(self):
        super().__init__()

        self.input_filenames = [None, None, None]
        self.output_filename = None

        self.setWindowTitle("Bias Correction Tool")
        self.setGeometry(300, 300, 550, 400)

        # Main window layout
        main_layout = QVBoxLayout(self)

        self.input_labels = []

        # ADCIRC File
        btn1 = QPushButton("Select ADCIRC Input File", self)
        btn1.clicked.connect(lambda checked, idx=0: self.select_input_file(idx))
        main_layout.addWidget(btn1)
        lbl1 = QLabel("Input 1: [Not Selected]", self)
        self.input_labels.append(lbl1)
        main_layout.addWidget(lbl1)

        # STWAVE File
        btn2 = QPushButton("Select STWAVE Input File", self)
        btn2.clicked.connect(lambda checked, idx=1: self.select_input_file(idx))
        main_layout.addWidget(btn2)
        lbl2 = QLabel("Input 2: [Not Selected]", self)
        self.input_labels.append(lbl2)
        main_layout.addWidget(lbl2)

        # ADCIRC Peaks File
        btn3 = QPushButton("Select ADCIRC Peaks File", self)
        btn3.clicked.connect(lambda checked, idx=2: self.select_input_file(idx))
        main_layout.addWidget(btn3)
        lbl3 = QLabel("Input 3: [Not Selected]", self)
        self.input_labels.append(lbl3)
        main_layout.addWidget(lbl3)

        # Output File
        self.btn_set_output = QPushButton("Set Output File", self)
        self.btn_set_output.clicked.connect(self.set_output_file)
        main_layout.addWidget(self.btn_set_output)
        self.lbl_output_path = QLabel("Output File: [Not Selected]", self)
        main_layout.addWidget(self.lbl_output_path)

        # Bias Parameters
        hbox_abs = QHBoxLayout()
        hbox_abs.addWidget(QLabel("Enter Absolute Bias:", self))
        self.abs_bias = QLineEdit(self)
        self.abs_bias.setText("1.0")
        hbox_abs.addWidget(self.abs_bias)
        main_layout.addLayout(hbox_abs)

        hbox_rel = QHBoxLayout()
        hbox_rel.addWidget(QLabel("Enter Relative Bias (Optional):", self))
        self.rel_bias = QLineEdit(self)
        self.rel_bias.setPlaceholderText("Optional (e.g., 0.5)")
        hbox_rel.addWidget(self.rel_bias)
        main_layout.addLayout(hbox_rel)

        # Resampling
        self.chk_resample = QCheckBox(
            "Resample time series at 1-hour increments?", self
        )
        main_layout.addWidget(self.chk_resample)

        # Process Button & Status
        self.btn_process = QPushButton("Process Files", self)
        self.btn_process.clicked.connect(self.process_files)
        self.btn_process.setEnabled(False)
        main_layout.addWidget(self.btn_process)

        self.lbl_status = QLabel("Status: Please provide all inputs.", self)
        main_layout.addWidget(self.lbl_status)

        self.setLayout(main_layout)

    def select_input_file(self, index):
        """Selects an input file for the given index (0, 1, or 2)."""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            f"Select Input File {index + 1}",
            "",
            "HDF5 Files (*.h5);;All Files (*)",
        )

        if file_name:
            self.input_filenames[index] = file_name
            self.input_labels[index].setText(f"Input {index + 1}: {file_name}")
            self.check_all_files_selected()

    def get_bias_values_from_mat(self):
        print("Get Bias Values From .mat")
        pass

    def set_output_file(self):
        """Sets the path for the single output file."""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Set Output File", "", "HDF5 Files (*.h5);;All Files (*)"
        )

        if file_name:
            self.output_filename = file_name
            self.lbl_output_path.setText(f"Output File: {file_name}")
            self.check_all_files_selected()

    def check_all_files_selected(self):
        """Checks if all files are selected and enables the Process button."""
        if all(self.input_filenames) and self.output_filename:
            self.btn_process.setEnabled(True)
            self.lbl_status.setText("Status: Ready to process.")
        else:
            self.btn_process.setEnabled(False)
            self.lbl_status.setText("Status: Please select all required files.")

    def process_files(self):
        """Reads from the 3 input files and writes their combined content to the output file."""
        # 1. Update status message and force UI repaint immediately
        self.lbl_status.setText("Status: Processing...")
        self.btn_process.setEnabled(
            False
        )  # Temporarily disable button to prevent double-clicks
        QApplication.processEvents()

        try:
            h5f1 = h5.File(self.input_filenames[0], "r")  # ADCIRC File
            h5f2 = h5.File(self.input_filenames[1], "r")  # STWAVE File
            h5f3 = h5.File(self.input_filenames[2], "r")  # ADCIRC Peaks File
            h5f4 = h5.File(self.output_filename, "w")
            storm_names = list(h5f2.keys())

            bias_a = float(self.abs_bias.text())
            rel_bias_text = self.rel_bias.text().strip()
            bias_r = float(rel_bias_text) if rel_bias_text else None

            should_resample = self.chk_resample.isChecked()

            attributes = h5f2.attrs
            nancount = 0

            # Dataset field names
            file_3a = "Water Elevation"
            file_3b = "yyyymmddHHMM"
            file_3c = "Mean Wave Direction"
            file_3d = "Mean Wave Period"
            file_3e = "Peak Period"
            file_3f = "Wind Direction"
            file_3g = "Wind Magnitude"
            file_3h = "Zero Moment Wave Height"

            for i, obj in enumerate(storm_names):
                groupadc = h5f1[obj]
                groupstw = h5f2[obj]
                grouppeaks = h5f3[obj]

                nans = math.isnan(groupstw["Water Elevation"][0])
                if nans:
                    nancount = nancount + 1
                else:
                    datesadc = [
                        datetime.strptime(str(int(x)), "%Y%m%d%H%M")
                        for x in groupadc["yyyymmddHHMM"]
                    ]
                    datesstw = [
                        datetime.strptime(str(int(x)), "%Y%m%d%H%M")
                        for x in groupstw["yyyymmddHHMM"]
                    ]

                    surge = groupadc["Water Elevation"][:]
                    peaksurge = grouppeaks["Water Elevation"][0]
                    surge[surge == max(surge)] = peaksurge

                    # Calculate bias adjustment
                    if bias_r is not None:
                        denom = (1.0 / bias_a**2) + (
                            1.0 / (bias_r * np.abs(surge)) ** 2
                        )
                    else:
                        denom = 1.0 / bias_a**2

                    biasadj = 1.0 / np.sqrt(denom)
                    surge = surge - (np.sign(bias_a) * biasadj)
                    # surge = np.subtract(surge, np.multiply(np.sign(bias_a), biasadj))

                    if should_resample:
                        peak_idx = np.argmax(surge)
                        t_peak = datesadc[peak_idx]
                        start_date = datesadc[0]
                        end_date = datesadc[-1]

                        before_peak = []
                        t_curr = t_peak - timedelta(hours=1)
                        while t_curr >= start_date:
                            before_peak.append(t_curr)
                            t_curr -= timedelta(hours=1)
                        before_peak.reverse()

                        after_peak = []
                        t_curr = t_peak + timedelta(hours=1)
                        while t_curr <= end_date:
                            after_peak.append(t_curr)
                            t_curr += timedelta(hours=1)

                        target_dates = before_peak + [t_peak] + after_peak

                        surge_resampled = self.interp_hydrograph(
                            surge, datesadc, target_dates
                        )
                        wavestoadcirc = self.interp_hydrograph(
                            groupstw["Zero Moment Wave Height"][:],
                            datesstw,
                            target_dates,
                        )

                        peak_target_idx = len(before_peak)
                        surge_resampled[peak_target_idx] = surge[peak_idx]

                        out_dates = np.array(
                            [int(d.strftime("%Y%m%d%H%M")) for d in target_dates]
                        )
                        out_surge = surge_resampled
                    else:
                        wavestoadcirc = self.interp_hydrograph(
                            groupstw["Zero Moment Wave Height"][:], datesstw, datesadc
                        )
                        out_dates = groupadc["yyyymmddHHMM"][:]
                        out_surge = surge

                    wavestoadcirc[wavestoadcirc < 0] = 0

                    grp2 = h5f4.create_group(obj)
                    dset3a = grp2.create_dataset(file_3a, data=out_surge)
                    dset3b = grp2.create_dataset(file_3b, data=out_dates)
                    dset3c = grp2.create_dataset(file_3c, data=np.zeros(len(dset3a)))
                    dset3d = grp2.create_dataset(file_3d, data=np.zeros(len(dset3a)))
                    dset3e = grp2.create_dataset(file_3e, data=np.zeros(len(dset3a)))
                    dset3f = grp2.create_dataset(file_3f, data=np.zeros(len(dset3a)))
                    dset3g = grp2.create_dataset(file_3g, data=np.zeros(len(dset3a)))
                    dset3h = grp2.create_dataset(file_3h, data=wavestoadcirc)

                    # Global attributes
                    h5f4.attrs["CHS Data Format"] = attributes[
                        "CHS Data Format"
                    ].decode("utf-8")
                    h5f4.attrs["Grid Name"] = "NACCS"
                    h5f4.attrs["Latitude Units"] = attributes["Latitude Units"].decode(
                        "utf-8"
                    )
                    h5f4.attrs["Longitude Units"] = attributes[
                        "Longitude Units"
                    ].decode("utf-8")
                    h5f4.attrs["Project"] = "Baltimore"
                    h5f4.attrs["Region"] = "Baltimore"
                    h5f4.attrs["Save Point ID"] = int(attributes["Save Point ID"])
                    h5f4.attrs["Save Point Latitude"] = float(
                        attributes["Save Point Latitude"]
                    )
                    h5f4.attrs["Save Point Longitude"] = float(
                        attributes["Save Point Longitude"]
                    )
                    h5f4.attrs["Vertical Datum"] = attributes["Vertical Datum"].decode(
                        "utf-8"
                    )

                    # Group-level attributes
                    if should_resample:
                        grp2.attrs["Record Interval"] = 1.0
                        grp2.attrs["Record Interval Units"] = "hour"
                    else:
                        grp2.attrs["Record Interval"] = groupadc.attrs[
                            "Record Interval"
                        ]
                        grp2.attrs["Record Interval Units"] = groupadc.attrs[
                            "Record Interval Units"
                        ].decode("utf-8")

                    grp2.attrs["Save Point Depth"] = float(
                        groupstw.attrs["Save Point Depth"]
                    )
                    grp2.attrs["Save Point Depth Units"] = groupstw.attrs[
                        "Save Point Depth Units"
                    ].decode("utf-8")
                    grp2.attrs["Steric Level"] = groupadc.attrs["Steric Level"].decode(
                        "utf-8"
                    )
                    grp2.attrs["Storm ID"] = groupstw.attrs["Storm ID"].decode("utf-8")
                    grp2.attrs["Storm Name"] = groupstw.attrs["Storm Name"].decode(
                        "utf-8"
                    )
                    grp2.attrs["Storm Type"] = "TS"

                    # Dataset attributes
                    dset3a.attrs["Model Variable"] = "eta"
                    dset3a.attrs["Units"] = "m"
                    dset3b.attrs["Units"] = "yyyymmddHHMM"
                    dset3c.attrs["Model Variable"] = "alpham"
                    dset3c.attrs["Units"] = "deg"
                    dset3d.attrs["Model Variable"] = "TM"
                    dset3d.attrs["Units"] = "sec"
                    dset3e.attrs["Model Variable"] = "Tp"
                    dset3e.attrs["Units"] = "sec"
                    dset3f.attrs["Model Variable"] = "UDIR"
                    dset3f.attrs["Units"] = "deg"
                    dset3g.attrs["Model Variable"] = "U"
                    dset3g.attrs["Units"] = "m/s"
                    dset3h.attrs["Model Variable"] = "Hmo"
                    dset3h.attrs["Units"] = "m"

            h5f1.close()
            h5f2.close()
            h5f3.close()
            h5f4.close()

            self.lbl_status.setText("Status: Processing complete!")

        except Exception as e:
            self.lbl_status.setText(f"Status: Error - {str(e)}")
            QMessageBox.critical(
                self, "Error", f"An error occurred during processing:\n{str(e)}"
            )
        finally:
            self.btn_process.setEnabled(True)

    def interp_hydrograph(self, y, t, tq):
        """
        Linear interpolation of signal values at query times.
        Parameters
        y : array-like
            Values of original signal
        t : array-like of datetime
            Time values of original signal
        tq : array-like of datetime
            Query time points

        Returns
        -------
        yq : np.ndarray
            Interpolated values at tq
        """  # Convert datetime arrays to numeric (seconds since t[0])
        t0 = t[0]
        t_sec = np.array([(ti - t0).total_seconds() for ti in t])
        tq_sec = np.array([(tqi - t0).total_seconds() for tqi in tq])

        # Build linear interpolator
        f = interp1d(t_sec, y, kind="linear", fill_value="extrapolate")

        # Evaluate at query points
        yq = f(tq_sec)
        return yq


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MultiInputFileApp()
    ex.show()
    sys.exit(app.exec_())

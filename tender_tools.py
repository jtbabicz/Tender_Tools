import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from scipy.optimize import curve_fit

class Tender:
    def __init__(self):
        pass
    
    def load_data(self, path, pattern, num_rows_to_skip):
        """
        Loads data from specified path and file name pattern creating a list of paths and a list of numpy array attributes.

        Parameters
        ----------
        path : str
            Path to data directory.
        pattern : str
            File name pattern.
        num_rows_to_skip : int
            Number of rows to skip if data file has a header.
        """
        self.data_path = glob(path + pattern)
        self.data_path.sort()
        self.data = [np.loadtxt(x, skiprows=num_rows_to_skip) for x in self.data_path]
        print("Loaded files:\n")
        for x in self.data_path:
            print(x)

    def normalize(self, measurement_type = "RIXS", method = "Area Integration"):
        """
        Intensity normalization of data. Area Integration method uses Numpy trapz method.

        Parameters
        ----------
        measurement_type : str
            Specifies which measurement type, either RIXS or XES
        method : str
            Integration method, e.g. Area Integration   
        """ 
        if measurement_type == "XES":
            if method == "Area Integration":
                norm_values = []
                data_norm = []
                for x in self.data:
                    xes_norm = np.trapz(x[:,1], x[:,0])
                    norm_values.append(xes_norm)

                    energy = x[:,0]
                    norm_xes = x[:,1] / xes_norm
                    data_norm.append(np.array([energy, norm_xes]).T)

                self.norm_values = norm_values
                self.data_norm = data_norm
            else:
                print("Error: Integration method not available")
                
        if measurement_type == "RIXS":
            if method == "Area Integration":
                norm_values = []
                data_norm = []
                for x in self.data:
                    herfd_norm = np.trapz(x[:,1], x[:,0]) 
                    tfy_norm = np.trapz(x[:,2], x[:,0])
                    norm_values.append([herfd_norm, tfy_norm])
        
                    energy = x[:,0]
                    norm_herfd = x[:,1] / herfd_norm
                    norm_tfy = x[:,2] / tfy_norm
                    data_norm.append(np.array([energy, norm_herfd, norm_tfy]).T)
                
                self.norm_values = norm_values
                self.data_norm = data_norm
            else:
                print("Error: Integration method not available")

    def combine_data(self, mode = None, incident_energies = None, indices = None):
        """
        Average all RIXS or XES data into single spectrum

        Parameters
        ----------
        mode : str
            Mode for combining data. Either RIXS or XES.
        incident_energies : list
            List of incident X-ray energies as string. Necessary to aggregrate and combine data.
        """
        self.incident_energies = incident_energies
        if mode == "RIXS" and incident_energies == None and indices == None:
            self.avg_data = np.mean(np.array(self.data), axis = 0)
            self.avg_data_norm = np.mean(np.array(self.data_norm), axis = 0)

        if mode == "RIXS" and incident_energies == None and type(indices) is list:

            temp_data = []
            temp_data_norm = []
            for x in indices:
                temp_data.append(self.data[x])
                temp_data_norm.append(self.data_norm[x])
                   
            self.avg_data = np.mean(np.array(temp_data), axis = 0)
            self.avg_data_norm = np.mean(np.array(temp_data_norm), axis = 0)

        if mode == "XES" and type(incident_energies) is list:
            self.indices_by_energy = []
            for energy in incident_energies:
                indices = []
                for x in range(len(self.data_path)):
                    if energy in self.data_path[x]:
                        indices.append(x)
                self.indices_by_energy.append(indices)

            self.list_avg_data = []
            self.list_avg_data_norm = []
            for x in self.indices_by_energy:
                print("Combining scans sorted by energy:")
                for y in x:
                    print(self.data_path[y])
                idx_min = np.min(x)
                idx_max = np.max(x) 

                avg_data = np.mean(np.array(self.data)[idx_min:idx_max + 1,:,:], axis = 0) ## idx_max needs to be increased by 1 since the slicing end value is EXCLUSIVE
                avg_norm_data = np.mean(np.array(self.data_norm)[idx_min:idx_max + 1,:,:], axis = 0) ## idx_max needs to be increased by 1 since the slicing end value is EXCLUSIVE
                
                self.list_avg_data.append(avg_data)
                self.list_avg_data_norm.append(avg_norm_data)

    def lorentzian(self, x, amp, center, width):
        y = amp * (width**2) / ((x - center)**2 + width**2)
        return y 

    def linear_model(self, x, m, b):
        y = m*x + b
        return y
        
    def fit_peaks(self, xlims=None):
        self.fit_results = []
        for x in self.list_avg_data_norm:
            popt, pcon = curve_fit(self.lorentzian, x[:,0], x[:,1], check_finite = True)
            print('Fit results: Amplitude = %.4f, Center = %7.2f, Width = %5.2f ' % tuple(popt))
            self.fit_results.append(dict(Parameters = popt, Convariance = pcon))
            
            plt.plot(x[:,0], self.lorentzian(x[:,0], *popt), 'r-', linewidth = 1, label = 'fit: A=%5.4f, C=%5.2f, W=%5.2f ' % tuple(popt))
            plt.scatter(x[:,0], x[:,1], label = "Data", s = 10, marker = 'o')
            plt.legend(bbox_to_anchor=(1, 1), fontsize=8)
            plt.xlabel("Pixel")
            plt.ylabel("Intensity (A.U.)")
            plt.xlim(xlims)
            plt.minorticks_on()

    def fit_energy_pixel(self, energies):
        e_calib = []
        for x, y in zip(self.fit_results, energies):
            print(x["Parameters"][1], y)
            e_calib.append([x["Parameters"][1], y])
        e_calib = np.array(e_calib)

        popt, pcov = curve_fit(self.linear_model, e_calib[:,0], e_calib[:,1])
        self.linear_fit = popt

        plt.scatter(e_calib[:,0], e_calib[:,1], label = 'Data', s = 70, marker = 'o')
        plt.plot(e_calib[:,0], self.linear_model(e_calib[:,0], *popt), 'r--', label = 'fit: m=%5.4f, b=%5.2f' % tuple(popt))
        plt.legend()
        plt.xlabel("Pixel")
        plt.ylabel("Energy (eV)")
        plt.minorticks_on()

    def make_energy_axis(self, x_axis):

        def conversion(x, m, b):
            return m*x + b

        m = self.linear_fit[0]
        b = self.linear_fit[1]

        self.energy_calibrated = conversion(x_axis, m, b)

    def plot(self, plot_type, data_type = "Norm", indices = [0],   
             fig_dim = (7, 7), label_range = [0, -1], y_increment = None, bbox_values = None,  
             xlims = None, title = None, xlabel = None, column=1, e_calibration = False,):
        
        plt.figure(figsize=fig_dim)

        if plot_type == "All_spectra":
            y = 0
            for x in indices:
                plt.plot(self.data_norm[x][:, 0], self.data_norm[x][:, column]+y, label = self.data_path[x][label_range[0]:label_range[1]])
                plt.xlabel(xlabel)
                plt.ylabel("Intensity")
                plt.xlim(xlims)
                plt.minorticks_on()  
                plt.legend(bbox_to_anchor=bbox_values, fontsize=8)                
                y += y_increment
            plt.title(title)
        if plot_type == "Final_XES":
            if e_calibration is None:
                y = 0
                for x in indices:
                    plt.subplot(2, 1, 1)
                    plt.plot(self.list_avg_data_norm[x][:, 0], self.list_avg_data_norm[x][:, 1], label = self.incident_energies[x])
                    plt.xlabel("Pixel")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()  
                    plt.legend()
                    plt.subplot(2, 1, 2)
                    plt.plot(self.list_avg_data_norm[x][:, 0], self.list_avg_data_norm[x][:, 1]+y, label = self.incident_energies[x])
                    plt.xlabel("Pixel")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()            
                    plt.legend() 
                    y += y_increment

            if type(e_calibration) is list:

                def conversion(x, m, b):
                    return m*x + b
                m = e_calibration[0]
                b = e_calibration[1]
                
                y = 0
                for x in indices:
                    x_ecalib = conversion(self.list_avg_data_norm[x][:,0], m, b)
                    self.x_ecalib_kb= x_ecalib
                    
                    plt.subplot(2, 1, 1)
                    plt.plot(x_ecalib, self.list_avg_data_norm[x][:, 1], label = self.incident_energies[x])
                    plt.xlabel("Energy (eV)")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()  
                    plt.legend()
                    plt.subplot(2, 1, 2)
                    plt.plot(x_ecalib, self.list_avg_data_norm[x][:, 1]+y, label = self.incident_energies[x])
                    plt.xlabel("Energy (eV)")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()            
                    plt.legend() 
                    y += y_increment

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
            
        if plot_type == "Final_RIXS":
            plt.subplot(2, 1, 1)
            plt.plot(self.avg_data_norm[:, 0], self.avg_data_norm[:, 1], label = "HERFD")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Intensity")
            plt.xlim(xlims)
            plt.minorticks_on()            
            plt.legend()    
            plt.subplot(2, 1, 2)
            plt.plot(self.avg_data_norm[:, 0], self.avg_data_norm[:, 2], label = "TFY")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Intensity")
            plt.xlim(xlims)
            plt.minorticks_on()
            plt.suptitle(title)
            plt.legend()    
        plt.tight_layout()

        if plot_type == "Stacked_HERFD_TFY":
            if data_type == "Norm":
                for x in indices:
                    plt.subplot(2, 1, 1)
                    plt.plot(self.data_norm[x][:,0], self.data_norm[x][:,1], label = self.data_path[x][label_range[0]:label_range[1]])
                    plt.legend(bbox_to_anchor=(1.2,1), fontsize=8)
                    plt.xlabel("Energy (eV)")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()
                    plt.subplot(2, 1, 2)
                    plt.plot(self.data_norm[x][:,0], self.data_norm[x][:,2], label = self.data_path[x][label_range[0]:label_range[1]])
                    plt.legend(bbox_to_anchor=(1.2,1), fontsize=8)
                    plt.xlabel("Energy (eV)")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()
                plt.suptitle("Normalized HERFD and TFY")
                plt.tight_layout()
                plt.show()

            if data_type == "Raw":
                for x in indices:
                    plt.subplot(2, 1, 1)
                    plt.plot(self.data[x][:,0], self.data[x][:,1], label = self.data_path[x][label_range[0]:label_range[1]])
                    plt.legend(bbox_to_anchor=(1.2,1), fontsize=8)
                    plt.xlabel("Energy (eV)")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()
                    plt.subplot(2, 1, 2)
                    plt.plot(self.data[x][:,0], self.data[x][:,2], label = self.data_path[x][label_range[0]:label_range[1]])
                    plt.legend(bbox_to_anchor=(1.2,1), fontsize=8)
                    plt.xlabel("Energy (eV)")
                    plt.ylabel("Intensity")
                    plt.xlim(xlims)
                    plt.minorticks_on()
                plt.suptitle("Raw HERFD and TFY")
                plt.tight_layout()
                plt.show()

    def save_data(self, filename, mode):
        if mode == "XES":
            if hasattr(self, 'list_avg_data_norm'):
                column_header = "Energy [eV]\tIntensity"
                for x, y in zip(self.incident_energies, self.list_avg_data_norm):
                    data = np.array([self.x_ecalib_kb, y[:,1]]).T
                    name = filename[:-4] + "_" + x + filename[-4:]
                    np.savetxt(name, data, header = column_header, delimiter = "\t")
                    print("Writing XES data to: " + str(name))
            else:
                print("Attribute: list_avg_data_norm is missing")
        elif mode == "RIXS":
            if hasattr(self, 'avg_data_norm'):
                column_header = "Energy [eV]\tHERFD\tTFY"
                np.savetxt(filename, self.avg_data_norm, header = column_header, delimiter = "\t")
                print("Writing RIXS Data to: " + str(filename))
            else:
                print("Attribute: list_avg_data_norm is missing")
        elif mode != "RIXS" or mode != "XES":
            print("Mode not recognized. Please specify XES or RIXS mode")


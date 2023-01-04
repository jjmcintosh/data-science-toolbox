#* DataToolbox.py 
#* 
#* Due on: May 2, 2022
#* Author(s): Jacob McIntosh 
#*



# inlcude dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import csv
import nltk
import os
import wordcloud



"""@package DataToolbox
ANLY 555 Spring 2022
Author(s): Jacob McIntosh 

This module encompasses a simple Data Science Toolbox for use in
importing data, implmenting simple algorithms and models, and 
assessing the accuracy of those algorithms and models
"""

class DataSet:
    """Abstract Base Class used by multiple subclasses for
    individual data types
    
    Holds data read in from a file for use in exploration and analysis
    
    Default Attributes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _DEFAULT_CLEAN_METHOD [str] = 'drop'; defines the default cleaning method to perform
    
    Supported Subclass Data Types
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    TimeSeriesDataSet: Time Series Data
    TextDataSet: Text Data
    QuantDataSubclass: Quantitative Data
    QualDataSet: Qualitative Data
    
    """
    
    
    _DEFAULT_CLEAN_METHOD = 'drop'
    
    
    def __init__(self, filename, header = 0):
        """Instantiates the DataSet class; invokes the load and readFromCSV class member functions
        
        Paramters
        ~~~~~~~~~
        filename [string]; The name of the file to load data in from
        header [int/None]; Whether or not the dataset has a header column. Default; first row of csv used as header
        
        
        Returns
        ~~~~~~~
        DataSet object 
        
        """
        
        self.__load(filename, header)
    
    def __readFromCSV(self, filename, header):
        """Private Member Function; Opens CSV file and returns it
        
        Paramters
        ~~~~~~~~~
        filename [string]; The name of the file to load data in from
        header [int/None]; Whether or not the dataset has a header column. Default; first row of csv used as header
        
        Returns
        ~~~~~~~
        Pandas Dataframe Object
        
        """
        try:
            return pd.read_csv(
                filepath_or_buffer = filename,
                header = header
            )
        except:
            raise FileNotFoundError('No such file or directory: \n' + os.getcwd() + '\\' + filename)
    
    def __load(self, filename, header):
        """Private Member Function; Opens and reads files
        
        Paramters
        ~~~~~~~~~
        filename [string]; The name of the file to load data in from
        header [int/None]; Whether or not the dataset has a header column. Default; first row of csv used as header
        
        Returns
        ~~~~~~~
        None
        
        """
        # load in data from CSV
        df = self.__readFromCSV(filename, header)
        
        # turn data into matrix
        self._data = df.to_numpy()
        
        # if matrix is non numeric
        if self._data.dtype.name == 'object':
            # cast matrix as string matrix
            self._data = self._data.astype(str)
        
        # get columnnames
        self._columnNames = df.columns.to_list()

    
    def _getColIdx(self, include = None, exclude = None):
        """Private Member Function; gets subset of column indexes
        
        Paramters
        ~~~~~~~~~
        include/exclude [list[index],list[str],int]; The indice(s) or column names to be included or excluded in subset. If include is declared, exclude will be ignored, Default: None, all columns will be returned
        
        Returns
        ~~~~~~~
        list[int]; column indexes
        
        """
        # if include/exclude is not specified, return all columns
        if include == None and exclude == None:
            colidx = [i for i in range(len(self._columnNames))]
        
        # if include is specified
        elif include != None:
            
            # if include is not a list, raise an error
            if type(include) == list:
                
               
            
                # if each element in list is an int
                if all(isinstance(col, int) for col in include):
                    # get column names by index
                    if all(np.array(include) < len(self._columnNames)):
                        colidx = include
                    # throw exceoption if index is out of range
                    else:
                        raise IndexError('Column index ' + str((np.array(include) < len(self._columnNames)).argmin()) + ' out of range')

                # if each element in list is a str
                elif all(isinstance(col, str) for col in include):
                    # for each element in inlcude
                    colidx = []
                    for col in include:
                        # confirm element is in the list of column
                        try:
                            colidx.append(self._columnNames.index(col))
                        
                        except:
                            # raise value error if not in column
                            raise ValueError(col + ' not in columns')
                    
                # otherwise, list is imcompatible, raise value error
                else:
                    raise TypeError('Include must of type [list[index],list[str],int]')
            
            
            # if include is a single integer index
            elif type(include) == int:
                    # get column names by index
                    if include < len(self._columnNames):
                        colidx = [include]
                    # throw exceoption of index is out of range
                    else:
                        raise IndexError('Column index ' + str(include) + ' out of range')
            else:
                raise TypeError('Include must of type [list[index],list[str],int]')
            
           

        # otherwise, exclude != None    
        else:

            # set colnames as list of columns
            colidx = [i for i in range(len(self._columnNames))]
            
            # if include is not a list, raise an error
            if type(exclude) == list:

                # if each element in list is an int
                if all(isinstance(col, int) for col in exclude):

                    # try to remove column names by indexes in exclude
                    try:
                        for col in exclude:
                            colidx.remove(col)
                    # throw exceoption of index is out of range
                    except:
                        raise IndexError('Column index ' + str(col) + ' out of range')

                # if each element in list is a str
                elif all(isinstance(col, str) for col in exclude):
                    # try to remove column names in exclude
                    try:
                        for col in exclude:
                            colidx.remove(self._columnNames.index(col))
                    # otherwise, an element in exclude is not in list of columns
                    except:
                            raise ValueError(col + ' not in columns')
                            
                # otherwise, list is imcompatible, raise value error
                else:
                    raise TypeError('Exclude must of type [list[index],list[str], int]')
            
            # if exlude is a single integer index
            elif type(exclude) == int:
                
                # get column names by index
                try:
                    colidx = colidx[:exclude] + colidx[exclude + 1:]
                # throw exceoption of index is out of range
                except:
                    raise IndexError('Column index ' + str(exclude) + ' out of range')
            
            else:
                raise TypeError('Exclude must of type [list[index],list[str], int]')

        
        # return list of columns
        return colidx
    
    
    def getColnames(self):
        """Member Function; returns list of column names

        Paramters
        ~~~~~~~~~
        None
        
        Returns
        ~~~~~~~
        list[str]; column names

        """
        return self._columnNames
    
    
    def head(self, n = 5):
        """Member Function; prints head of data

        Paramters
        ~~~~~~~~~
        n [int]; The number of rows to print
        
        Returns
        ~~~~~~~
        None

        """
        
        # if n not int, raise error
        if type(n) != int:
            raise TypeError('n must be of type int; got ' + str(type(n)))
        
        # if too many columns to print
        if len(self._columnNames) > 6:

            # get subset of columns and print
            for col in self._columnNames[:3] + ['...'] + self._columnNames[-3:]:
                print(str(col).ljust(15), end = '')
            
            # divider line
            print('\n'+''.join(['*' for i in range(120)]))
            
            # for each row of subsetted data
            for row in np.hstack(
                (
                    self._data[:n,:3],
                    np.array([[''] for i in range(n)]),
                    self._data[:n,-3:]
                )
            ):
                # for each value in row, print with alignment
                for col in row:
                    print(str(col)[:10].ljust(15), end = '')
                # new line
                print()
        
        # otherise, enough columns to print
        else:
            
            # for each column, print column name
            for col in self._columnNames:
                print(str(col).ljust(45), end = '')
            
            # divder line
            print('\n'+''.join(['*' for i in range(120)]))
            
            # for each row in subsetted data
            for row in self._data[:n,:]:
                # for each value in row, print with alignment
                for col in row:
                    print(str(col)[:40].ljust(45), end = '')
                print()

                
                
    def tail(self, n = 5):
        """Member Function; prints tail of data

        Paramters
        ~~~~~~~~~
        n [int]; The number of rows to print
        
        Returns
        ~~~~~~~
        None

        """
        
        # if n not int, raise error
        if type(n) != int:
            raise TypeError('n must be of type int; got ' + str(type(n)))
        
        # if too many columns to print
        if len(self._columnNames) > 6:

            # get subset of columns and print
            for col in self._columnNames[:3] + ['...'] + self._columnNames[-3:]:
                print(str(col).ljust(15), end = '')
            
            # divider line
            print('\n'+''.join(['*' for i in range(110)]))
            
            # for each row of subsetted data
            for row in np.hstack(
                (
                    self._data[-n:,:3],
                    np.array([[''] for i in range(n)]),
                    self._data[-n:,-3:]
                )
            ):
                # for each value in row, print with alignment
                for col in row:
                    print(str(col)[:10].ljust(15), end = '')
                # new line
                print()
        
        # otherise, enough columns to print
        else:
            
            # for each column, print column name
            for col in self._columnNames:
                print(str(col).ljust(45), end = '')
            
            # divder line
            print('\n'+''.join(['*' for i in range(110)]))
            
            # for each row in subsetted data
            for row in self._data[-n:,:]:
                # for each value in row, print with alignment
                for col in row:
                    print(str(col)[:40].ljust(45), end = '')
                print()
    
    
    def getData(self):
        """Member Function; returns the dataset in matrix format

        Paramters
        ~~~~~~~~~
        None
        
        Returns
        ~~~~~~~
        _data [np.arr]; Numpy Array of dataset

        """
        return self._data
    
    
    def getDefaultCleanMethod(self):
        """Member Function; returns the default cleaning method

        Paramters
        ~~~~~~~~~
        None
        
        Returns
        ~~~~~~~
        _DEFAULT_CLEAN_METHOD [str]; default cleaning method

        
        """
        return self._DEFAULT_CLEAN_METHOD
    

    def setDefaultCleanMethod(self, method):
        """Member Function; updates the default cleaning method

        Paramters
        ~~~~~~~~~
        method [str]; new default cleaning method
        
        Returns
        ~~~~~~~
        None

        
        """
        
        self._DEFAULT_CLEAN_METHOD = method
        
        pass
    
    
    def clean(self, include = None, exclude = None, method = None):
        """Member Function; performs a cleaning routine on data

        Paramters
        ~~~~~~~~~
        include/exclude [list[index],list[str],int]; The indice(s) or column names to be included or excluded from cleaning process. If include is declared, exclude will be ignored, Default: None, all columns will be cleaned
        method [str]; Describes the operation to address NA values. Must be in {'drop', 'mean', 'median', 'mode'}
        
        Returns
        ~~~~~~~
        None

        """
        
        # get list of columns to clean
        colidx = self._getColIdx(
            include = include,
            exclude = exclude
        )
            
        # if method is not specified, use default class cleaning method
        if method == None:
            method = self._DEFAULT_CLEAN_METHOD
        
        # if 'drop', drop all rows with missing values in columns in colidx
        if method == 'drop':
            
            # if data is numeric
            if self._data.dtype.char == 'd':
            
                # drop rows
                for idx in colidx:
                    self._data =  self._data[~np.isnan(self._data[:,idx]),:]
            
             # otherwise data is text
            else:
                
                # drop rows
                for idx in colidx:
                    self._data =  self._data[(self._data[:,idx] != 'nan'),:]
            

            
        # if 'mean', replace all missing values with column mean for each column in colnames           
        elif method == 'mean':
            
             # if data is not numeric
            if self._data.dtype.char != 'd':
                raise TypeError('Columns must numeric to perfom method')
            
             # replace with mean
            for idx in colidx:
                self._data[np.isnan(self._data[:,idx]),idx] = np.nanmean(self._data[:,idx])
            
                
        # if 'median', replace all missing values with column median for each column in colnames            
        elif method == 'median':            
            
             # if data is not numeric
            if self._data.dtype.char != 'd':
                raise TypeError('Columns must numeric to perfom method')
            
             # replace with median
            for idx in colidx:
                self._data[np.isnan(self._data[:,idx]),idx] = np.nanmedian(self._data[:,idx])
                
        # if 'mode', replace all missing values with column mode for each column in colnames            
        elif method == 'mode':
            
             # if data is numeric
            if self._data.dtype.char == 'd':
                
                # for each index in colidx
                for idx in colidx:
                    
                    # get value counts
                    freq_count = np.unique(
                        self._data[:,idx],
                        return_counts=True
                    )
                    
                    # replace with mode
                    self._data[np.isnan(self._data[:,idx]),idx]  = freq_count[0][freq_count[1].argmax()]
            
            # otherwise data is text
            else:
                # for each index in colidx
                for idx in colidx:
                    
                    # get value counts
                    freq_count = np.unique(
                        self._data[self._data[:,idx] != 'nan',idx],
                        return_counts=True
                    )
                    
                    # replace nan with mode
                    self._data[(self._data[:,idx] == 'nan'),idx] = freq_count[0][freq_count[1].argmax()]

        # otherwise, cleaning method not defined, raise error
        else:
            raise ValueError('Cleaning Method \'' + method + '\' not defined')


    def explore(self, x):
        """Member Function; Displays histogram or word count frequency based on data type

        Paramters
        ~~~~~~~~~
        x [list[index],list[str],int]; The single indice or column name to be used for visualization
        
        Returns
        ~~~~~~~
        None
        
        """
        
        # get x column
        x = self._getColIdx(
            include = x
        )
        
        # if numeric, plot histogram
        if self._data.dtype.char == 'd':
            fig, ax = plt.subplots(figsize = (15,10))

            # plot histogram
            ax.hist(
                x = self._data[:,x])

            # set x label
            ax.set_xlabel(
                self._columnNames[x[0]],
                fontsize = 20
            )

            # set y label
            ax.set_ylabel(
                'Count',
                fontsize = 20
            )

            # set tick label size
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = 15
            )

            # set title
            ax.set_title("Histogram of " + str(self._columnNames[x[0]]), fontsize = 30)

            # show plot
            plt.show()
        
        
        #otherise, string column
        else:
            
            # concat text
            text = ' '.join(self._data[:,x].flatten())
            
            # split text on spaces, then get value counts for each word
            text = np.unique(np.array(text.split(' ')), return_counts= True)
            
            
            # get sorted index of word frequencies
            order = np.flip(np.argsort(text[1]))
            
            # reshape and stack array into 2D array
            text = np.hstack((text[0][order].reshape(-1,1), text[1][order].reshape(-1,1)))
            
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))
            
            # plot bar plot of word frequencies
            ax.bar(
                x = text[:10,0].flatten(),
                height = text[:10,1].flatten().astype(int)
            )
        
            
            # set x and y labels
            ax.set_xlabel(self._columnNames[x[0]], fontsize = 20)
            ax.set_ylabel('Frequency', fontsize = 20)
            
            # set tick label size
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = 15
            )

            
            # set title
            ax.set_title(
                "Word Count Frequency",
                fontsize = 30
            )

            # show plot
            plt.show()
        
        
        pass

    
class TimeSeriesDataSet(DataSet):
    """Subclass of DataSet Abstract Base Class; Supports and holds Time Series Data
 
    """
    def __init__(self, filename):
        """Instantiates the TimeSeriesDataSet class; inherits member functions and attributes from DataSet base class
        
        Paramters
        ~~~~~~~~~
        filename [string]; The name of the file to load data in from
        
        
        Returns
        ~~~~~~~
        TimeSeriesDataSet [object] 
        
        """
        super().__init__(filename)
        
        
    def clean(self, filter_width = 5, include = None, exclude = None):
        """Member Function; Overridden from SuperClass; Replaces all missing values with a median filter

        Paramters
        ~~~~~~~~~
        filter_width [int]; width of filter for median filter replacement
        include/exclude [list[index],list[str],int]; The indice(s) or column names to be included or excluded from cleaning process. If include is declared, exclude will be ignored, Default: None, all columns will be cleaned
  
        Returns
        ~~~~~~~
        None

        """
        # check if filter_width is of type int, raise exception otherwise
        if type(filter_width) != int:
            raise TypeError('filter_width must be of type int; got ' + str(type(filter_width)))
        
        
        # get list of columns to clean
        colidx = self._getColIdx(
            include = include,
            exclude = exclude
        )
        
        # for each row of data
        for idx in range(self._data.shape[0]):
            # fill missing values of each row with the median of each filter defined by filterwidth on columns in colnames
            
    
            
            # assign median filter to missing values in idx row and columns in colidx
            self._data[idx, np.isnan(self._data[idx,:]) & [i in colidx for i in range(self._data.shape[1])]] = np.nanmedian(

                # get filter of data centered around row (i) for each column in colidx
                self._data[
                    # lower bound
                    (idx-filter_width if idx-filter_width >= 0 else 0) :
                    #upper bound
                    (idx + filter_width + 1 if idx + filter_width + 1 < self._data.shape[0] else self._data.shape[0] - 1),

                    #column indexs to subset
                    :
                    # get median of each column
                ],
                # along the columns
                axis = 0
            # indexed by the columns that have a missing value in the row
            )[np.isnan(self._data[idx,:]) & [i in colidx for i in range(self._data.shape[1])]]
        
        pass
    
    def explore(self, x, y, method = 'line'):
        """Member Function; Overridden from SuperClass; Plots either a line plot or a summary statistics plot

        Paramters
        ~~~~~~~~~
        x [list[index],list[str],int]; The single indice or column name to be used for the x-axis
        y [list[index],list[str],int]; The indices or column names to be used for the y-axis
        method [str]; The type of plot to show. Must be 'line' or 'summary'. Default: 'line'
        
        Returns
        ~~~~~~~
        None
        
        """
        
        
        # get x column
        x = self._getColIdx(
            include = x
        )
        # if x is not a single column, raise error
        if len(x) != 1:
            raise ValueError('x must be a single column; got ' + str(len(x)) + ' columns')
        
        # if y is not provided, raise error
        if y == None:
            raise ValueError('y has not been declared. Too few parameters')
        
        # get y column
        y = self._getColIdx(
            include = y
        )
        
        # if method is line, perform line plot
        if method == 'line':
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))
            

            # get color map
            cmap = matplotlib.cm.get_cmap('jet')

            # factor to normalize colormap to scale of discrete categories
            norm_factor = len(y)-1

            # scatter plot for each hue (category)
            for _,col in enumerate(y):
                ax.plot(
                    self._data[:,x],
                    self._data[:,col],
                    color = cmap((_/norm_factor if norm_factor > 0 else 0)),
                    label = str(self._columnNames[col]),
                    linewidth = 3,
                    alpha = 0.6
                    )
            # plot legend    
            ax.legend(
                loc = 'center right',
                bbox_to_anchor = (1.15, 0.5),
                fontsize = 20,
            )
            
            
            #set x and y limits
            ax.set_xlim(np.min(self._data[:,x]), np.max(self._data[:,x]) )
            ax.set_ylim(np.min(self._data[:,y]), np.max(self._data[:,y]) )
            
            # set x and y labels
            ax.set_xlabel(self._columnNames[x[0]], fontsize = 20)
            
            # set tick label size
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = 15
            )

            
            # set title
            ax.set_title(
                "Line Plot",
                fontsize = 30
            )

            # show plot
            plt.show()
        
        # otherwise if method is summary, plot the summary plot
        elif method == 'summary':
            
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))
            
            # compute standard deviation across columns
            std = np.std(
                self._data[:,y],
                axis = 1
            )
            
            # compute standard deviation across columns
            mean = np.mean(
                self._data[:,y],
                axis = 1
            )

            # plot mean values over time
            ax.plot(
                self._data[:,x],
                mean,
                linewidth = 3,
                label = "mean"
                )
            
            # plot std errorbars
            plt.errorbar(
                self._data[:,x],
                mean,
                yerr = std,
                fmt = 'none',
                label = "std",
                linewidth = 1
                )
            
            # plot legend    
            ax.legend(
                loc = 'center right',
                bbox_to_anchor = (1.2, 0.5),
                fontsize = 20,
            )
            
            
            #set x and y limits
            ax.set_xlim(np.min(self._data[:,x]), np.max(self._data[:,x]) )
            ax.set_ylim(max(0,np.min(mean - 2*std)), np.max(mean + 2*std) )
            
            # set x and y labels
            ax.set_xlabel(self._columnNames[x[0]], fontsize = 20)
            
            # set tick label size
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = 15
            )

            
            # set title
            ax.set_title(
                "Summary Plot",
                fontsize = 30
            )

            # show plot
            plt.show()
            
            
        # otherwise, method not defined, throw error
        else:
            raise ValueError("Method must be 'line' or 'summary'; got " + method)
            


class TextDataSet(DataSet):
    """Subclass of DataSet Abstract Base Class; Supports and holds Text data
 
    """
    def __init__(self, filename):
        """Instantiates the TextDataSet class; inherits member functions and attributes from DataSet base class
        
        Paramters
        ~~~~~~~~~
        filename [string]; The name of the file to load data in from
        
        
        Returns
        ~~~~~~~
        TextDataSet [object] 
        
        """
        # call super constructor
        super( ).__init__(filename)
        
        
        # assign stopwords 
        self._stopwords = nltk.corpus.stopwords.words('english')

        
     
    
    def getStopwords(self):
        """Member Function; retrieves the list of NLTK stopwords

        Paramters
        ~~~~~~~~~
        None
        
        Returns
        ~~~~~~~
        list[str]; list of stopwords

        """
        return self._stopwords
    
    
    def setStopwords(self, stopwords):
        """Member Function; updates the list of stopwords

        Paramters
        ~~~~~~~~~
        list[str]; list of stopwords
        
        Returns
        ~~~~~~~
        None

        """
        # assign stopwords
        self._stopwords = stopwords
    
    
    
    
    def clean(self, include = None, exclude = None, stem = True):
        """Member Function; Overridden from SuperClass; Removes stop words from text data with the option to stem the data

        Paramters
        ~~~~~~~~~
        include/exclude [list[index],list[str]]; The indices or column names to be included or excluded from cleaning process. If include is declared, exclude will be ignored, Default: None, all columns will be cleaned
        stem [bool]; Whether or not to stem the tokens in the documents. Default: True
        
        Returns
        ~~~~~~~
        None

        """
        
        # get list of columns to clean
        colidx = self._getColIdx(
            include = include,
            exclude = exclude
        )
        
        # intialize stemmer and tokenizer
        stemmer = nltk.SnowballStemmer(language = 'english')
        tokenizer = nltk.RegexpTokenizer(pattern = r'\w+')
        
        # if stem is true
        if stem:
            # for each column
            for col in colidx:

                try:
                    # tokenize, remove stopwords, make lowercase, and stem document
                    # then join on ' '
                    self._data[:, col] = np.array(
                        list(
                            map(
                                lambda x:
                                    ' '.join(
                                        [stemmer.stem(token.lower()) for token in  tokenizer.tokenize(x) if token not in self._stopwords]
                                ),
                                self._data[:, col]
                            )
                        )
                    )
                    
                # column is not of type string, raise error    
                except:
                    raise TypeError("Expected column of type string")
        
        # otherwise, don't stem
        else:
            
            # for each column
            for col in colidx:

                try:
                    # tokenize, remove stopwords, make lowercase
                    # then join on ' '
                    self._data[:, col] = np.array(
                        list(
                            map(
                                lambda x:
                                    ' '.join(
                                        [token.lower() for token in  tokenizer.tokenize(x) if token not in self._stopwords]
                                ),
                                self._data[:, col]
                            )
                        )
                    )
                    
                # column is not of type string, raise error    
                except:
                    raise TypeError("Expected column of type string")
    
    def explore(self, x, method = 'wordcloud'):
        """Member Function; Overridden from SuperClass; Plots either a wordcloud or a wordcount frequency plot

        Paramters
        ~~~~~~~~~
        x [list[index],list[str],int]; The single indice or column name to be used for the visualization
        method [str]; The type of plot to show. Must be 'wordcloud' or 'freq'. Default: 'wordcloud'
        
        
        Returns
        ~~~~~~~
        None
        
        """
        
        # get x column
        x = self._getColIdx(
            include = x
        )
        
        # if x is not a single column, raise error
        if len(x) != 1:
            raise ValueError('x must be a single column; got ' + str(len(x)) + ' columns')
        
        
        # concat text
        text = ' '.join(self._data[:,x].flatten())
        
        # if method is wordcloud
        if method == 'wordcloud':
            
            # generate wordcloud object
            wc= wordcloud.WordCloud(
                background_color='white',
                max_font_size = 50).generate(text)
            
            # set up plot
            plt.figure(figsize = (15,10))
            
            # push wordcloud object on figure
            plt.imshow(wc)
            
            # turn off axis
            plt.axis('off')
            
            # show plot
            plt.show()
            
        # otherwise, if method is freq
        elif method == 'freq':
            
            # split text on spaces, then get value counts for each word
            text = np.unique(np.array(text.split(' ')), return_counts= True)
            
            # get sorted index of word frequencies
            order = np.flip(np.argsort(text[1]))
            
            # reshape and stack array into 2D array
            text = np.hstack((text[0][order].reshape(-1,1), text[1][order].reshape(-1,1)))
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))
            
            # plot bar plot of word frequencies
            ax.bar(
                x = text[:10,0].flatten(),
                height = text[:10,1].flatten().astype(int)
            )
        
            # set x and y labels
            ax.set_xlabel(self._columnNames[x[0]], fontsize = 20)
            ax.set_ylabel('Frequency', fontsize = 20)
            
            # set tick label size
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = 15
            )

            # set title
            ax.set_title(
                "Word Count Frequency",
                fontsize = 30
            )

            # show plot
            plt.show()
            
        # otherwise, method not defined, throw error
        else:
            raise ValueError("Method must be 'wordcloud' or 'freq'; got " + method)
        
        pass
    
        
class QuantDataSet(DataSet):
    """Subclass of DataSet Abstract Base Class; Supports and holds quantitative Data
    
    Default Attributes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _DEFAULT_CLEAN_METHOD [str] = 'mean'; defines the default cleaning method to perform
    
    
    """

    _DEFAULT_CLEAN_METHOD = 'mean'
    
    def __init__(self, filename):
        """Instantiates the QuantDataSet class; inherits member functions and attributes from DataSet base class
        
        Paramters
        ~~~~~~~~~
        filename [string]; The name of the file to load data in from
        
        
        Returns
        ~~~~~~~
        QuantDataSet [object] 
        
        """
        
        super( ).__init__(filename, header= None)
        

    
    def explore(self, x , y = None , hue = None, method = 'hist'):
        """Member Function; Overridden from SuperClass; plots either a scatterplot or a histogram

        Paramters
        ~~~~~~~~~
        x [list[index],list[str],int]; The single indice or column name to be used for the x-axis
        y [list[index],list[str],int]; The single indice or column name to be used for the y-axis; Required only for 'scatter' method
        hue [list[index],list[str],int]; The single indice or column name to aggregate over x and y columns; Optional parameter for 'scatter' method
        method [str]; The type of plot to show. Must be 'scatter' or 'hist'. Default: 'hist'
        
        Returns
        ~~~~~~~
        None
        
        """
        
        # get x column
        x = self._getColIdx(
            include = x
        )
        # if x is not a single column, raise error
        if len(x) != 1:
            raise ValueError('x must be a single column; got ' + str(len(x)) + ' columns')
        
        if method == 'scatter':
            # if y is not provided, raise error
            if y == None:
                raise ValueError('y has not been declared. Too few parameters')
            
            # get y column
            y = self._getColIdx(
                include = y
            )
            # if y is not a single column
            if len(y) != 1:
                raise ValueError('y must be a single column; got ' + str(len(x)) + ' columns')
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))
            
            # if hue is provided
            if hue != None:
                
                # get hue column
                hue = self._getColIdx(
                    include = hue
                )
                # if hue is not a single column
                if len(hue) != 1:
                    raise ValueError('hue must be a single column; got ' + str(len(x)) + ' columns')
                
                # get color map
                cmap = matplotlib.cm.get_cmap('viridis')
                
                # factor to normalize colormap to scale of discrete categories
                norm_factor = len(np.unique(self._data[:,hue])) -1

                # scatter plot for each hue (category)
                for _,lab in enumerate(np.unique(self._data[:,hue])):
                    ax.scatter(
                        x = self._data[(self._data[:,hue] == lab).flatten(),x],
                        y = self._data[(self._data[:,hue] == lab).flatten(),y],
                        color = cmap((_/norm_factor if norm_factor > 0 else 0)),
                        label = str(lab),
                        alpha = 0.4
                        )
                # plot legend    
                ax.legend(
                    loc = 'center right',
                    bbox_to_anchor = (1.15, 0.5),
                    fontsize = 20,
                    title = self._columnNames[hue[0]], 
                    title_fontsize = 20
                )
            
            # otherwise, hue is not provided
            else:
                # plot scatterplot
                ax.scatter(
                    x = self._data[:,x],
                    y = self._data[:,y],
                    alpha = 0.4
                    )    
            
            #set x and y limits
            ax.set_xlim(np.min(self._data[:,x]), np.max(self._data[:,x]) )
            ax.set_ylim(np.min(self._data[:,y]), np.max(self._data[:,y]) )
            
            # set x and y labels
            ax.set_xlabel(self._columnNames[x[0]], fontsize = 20)
            ax.set_ylabel(self._columnNames[y[0]], fontsize = 20)
            
            # set tick label size
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = 15
            )

            
            # set title
            ax.set_title(
                "Scatterplot",
                fontsize = 30
            )

            # show plot
            plt.show()
            
        elif method == 'hist':
            fig, ax = plt.subplots(figsize = (15,10))

            # plot histogram
            ax.hist(
                x = self._data[:,x])

            # set x label
            ax.set_xlabel(
                self._columnNames[x[0]],
                fontsize = 20
            )
            
            # set y label
            ax.set_ylabel(
                'Count',
                fontsize = 20
            )

            # set tick label size
            ax.tick_params(
                axis = 'both',
                which = 'both',
                labelsize = 15
            )

            # set title
            ax.set_title("Histogram of " + str(self._columnNames[x[0]]), fontsize = 30)

            # show plot
            plt.show()
            
        # otherwise, method not defined, throw error
        else:
            raise ValueError("Method must be 'hist' or 'scatter'; got " + method)
            
    
    
class QualDataSet(DataSet):
    """Subclass of DataSet Abstract Base Class; Supports and holds qualitative Data
 
    Default Attributes
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _DEFAULT_CLEAN_METHOD [str] = 'mode';  defines the default cleaning method to perform
    
    """
    
    
    _DEFAULT_CLEAN_METHOD = 'mode'
    
    def __init__(self, filename):
        """Instantiates the QualDataSet class; inherits member functions and attributes from DataSet base class
        
        Paramters
        ~~~~~~~~~
        filename [string]; The name of the file to load data in from
        
        
        Returns
        ~~~~~~~
        QualDataSet [object] 
        
        """
        super( ).__init__(filename)
        
    
    def explore(self, x, hue = None, method = 'stacked'):
        """Member Function; Overridden from SuperClass; plots either a stacked bar chart or a proportion chart

        Paramters
        ~~~~~~~~~
        x [list[index],list[str],int]; The single indice or column name to be used for the x-axis
        hue [list[index],list[str],int]; The single indice or column name to aggregate the x column by; Optional parameter for 'stacked' method
        method [str]; The type of plot to show. Must be 'stacked' or 'prop'. Default: 'stacked'
        
        Returns
        ~~~~~~~
        None
        
        """
        
        # get x column
        x = self._getColIdx(
            include = x
        )
        
        # if x isnt a single column, raise error
        if len(x) != 1:
            raise ValueError('x must be a single column; got ' + str(len(x)) + ' columns')
        
        # if method is stacked bar plot
        if method == 'stacked':
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))
            
            # get labels of column
            labs = np.unique(self._data[:,[x]])

            # if hue is provided
            if hue != None:
                
                
                # get hue column
                hue = self._getColIdx(
                    include = hue
                )
                # if hue is not a single column
                if len(hue) != 1:
                    raise ValueError('hue must be a single column; got ' + str(len(x)) + ' columns')
                
                # intitilize array of zeros with length equal to number of categories
                bottom = np.zeros(len(np.unique(self._data[:,x])))

                # for each unique category in hue
                for val in np.unique(self._data[:,hue]):
                    
                    # plot bar chart
                    ax.bar(
                        labs,
                        np.unique(self._data[(self._data[:,hue] == val).flatten(),x], return_counts = True)[1],
                        width = 0.7,
                        bottom = bottom,
                        label=val
                    )


                    # update where plots start plotting
                    bottom += np.unique(self._data[(self._data[:,hue] == val).flatten(),x], return_counts = True)[1]

                # plot legend
                ax.legend(
                    fontsize = 15,
                    title = self._columnNames[hue[0]],
                    title_fontsize = 20 
                )
                # set title
                ax.set_title(
                    'Freqeuncy of ' + self._columnNames[x[0]] + ' by ' + self._columnNames[hue[0]],
                    fontsize = 25
                )

            # otherwise, hue is not provided
            else:
                # bar plot
                ax.bar(
                        labs,
                        np.unique(self._data[:,x], return_counts = True)[1],
                        width = 0.7
                    )
                # set title
                ax.set_title('Freqeuncy of ' + self._columnNames[x[0]], fontsize = 25)
            
            # update tick labels
            ax.tick_params(
                axis = 'both',
                labelsize = 15)
            
            # set x and y labels
            ax.set_ylabel('Frequency', fontsize = 20)
            ax.set_xlabel(self._columnNames[x[0]], fontsize = 20)


            # show plot
            plt.show()
        
        # otherwsie if its the proportion method
        elif method == 'prop':
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))

            # calculate widiths
            widths = np.unique(self._data[:,[x]], return_counts= True)[1]
            widths = widths/sum(widths)


            # get color map
            cmap = matplotlib.cm.get_cmap('jet')

            # factor to normalize colormap to scale of discrete categories
            norm_factor = len(widths) -1

            # get dummy variables
            left = np.zeros(1)
            labs = ['temp'] 

            # for each width, plot horizontal bar
            for _,val in enumerate(widths):

                ax.barh(
                    y = labs,
                    height = 1,
                    width = val,
                    label = val,
                    color = cmap((_/norm_factor if norm_factor > 0 else 0)),
                    left = left
                )

                left += val

            # set axis limits and turn off axes
            ax.set_xlim(0,1)
            ax.set_ylim(-0.5,0.5)
            ax.axis('off')

            # include legend
            ax.legend(
                labels = np.unique(self._data[:,[x]]),
                title = self._columnNames[x[0]],
                title_fontsize = 20,
                fontsize = 15,
                loc = 'center right',
                bbox_to_anchor = (1.2,0.5)
            )

            # set title
            ax.set_title(
                "Proportion Plot of " + self._columnNames[x[0]],
                fontsize = 25
            )

            # show plot
            plt.show()

        # otherwise, method not defined, throw error
        else:
            raise ValueError("Method must be 'stacked' or 'prop'; got " + method)


# In[1]:


class ClassifierAlgorithm:
    """Abstract Base Class used by multiple subclasses for different algorithms
    
    Performs K-Nearest Neighbor Algorithms on data
    
    Supported Subclass Algorithms
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    simpleKNNClassifier: Simple KNN Algorithm
    kdTreeKNNClassifier: k-Dimensional Tree KNN Algorithm
    
    
    Default Attributes
    ~~~~~~~~~~~~~~~
    
    _data [None]: Holds the training data of the Classifier; when initialized in train(), will hold a numpy.ndarray of shape (n,m)
    _labels [None]: Holds the training labels of the Classifier; When initialized in train(), will hold a numpy.ndarray of shape (n,)
    _preds [None]: Holds the test prediction labels of the Classifier; When initialized in test(), will hold a numpy.ndarray of shape (j,)
    
    """
    
    # Default Attributes
    _data = None
    _labels = None
    _preds = None
    
    def __init__(self):
        """Instantiates the ClassifierAlgorithm class, invokes initialization of Default protected attributes
        
        Paramters
        ~~~~~~~~~
        None
        
        
        Returns
        ~~~~~~~
        ClassifierAlgorithm [object] 
        
        """

        
        
    def train(self, data, labels):
        """ Member Function; trains the classifier on the imported data
 
        
        Paramters
        ~~~~~~~~~
        data [numpy.ndarray]: an array of shape (m,n) that holds the training data
        labels [numpy.ndarray]: an array of shape (m,) that holds the training labels
        
        
        Returns
        ~~~~~~~
        None
        
        """
        
        # if data or labels are not numpy.ndarrays, raise an error
        if not isinstance(data, np.ndarray) or not isinstance(labels, np.ndarray):
            raise TypeError(
                'Data and Labels must be type numpy.ndarray with shapes (n,m) and (n,) ; got ' + str(type(data)) + 'and ' + str(type(labels))
            )
        
        # if there is a dimension mismatch along the columns
        if len(data) != len(labels):
            raise ValueError(
                'Dimension Mismatch: Data and Labels must be numpy.ndarray with shapes (n,m) and (n,); got ' + str(data.shape) + 'and ' + str(labels.shape)
            )
        
        # shallow copies of data and labels
        self._data = data
        self._labels = labels.flatten()
        self._uniquelabels = len(set(self._labels))
        
        
        
        pass
    
    def test(self):
        """Member Function; tests the classifier on the imported data; Abstract method; does not operate
 
        
        Paramters
        ~~~~~~~~~
        None
        
        
        Returns
        ~~~~~~~
        None
        
        """
        pass    
    

class simpleKNNClassifier(ClassifierAlgorithm):
    """Subclass of ClassifierAlgorithm Abstract Base Class; performs simple KNN algorithm
 
    """
    

    
    def __init__(self, k = 10, dist_method = 'euclidean'):
        """Instantiates the simpleKNNClassifier class;
        inherits member functions and attributes from ClassifierAlgorithm base class
        
        Paramters
        ~~~~~~~~~
        k [int]: A postive integer dictating the number of neighbors: Default: 10
        dist_method [str]: the distance metric to use; must be 'euclidean' or 'manhattan'; Default: 'euclidean'
        
        Returns
        ~~~~~~~
        simpleKNNClassifier [object] 
        
        """
        
        # if k isn't a positive integer, raise error
        if  not isinstance(k, int) or k < 1:
            raise ValueError(
                'k must be a positive integer'
            )
        
        # if dist_method isnt defined
        if dist_method not in ['euclidean', 'manhattan']:
            raise ValueError(
                "distance must be 'euclidean' or 'manhattan'; got " + str(distance)
            )
        
        # assign subclass specfic attributes
        self._k = k    
        self._dist_method = dist_method
        
        # call super constructor
        super( ).__init__()

    
    # Function call
    # Time Complexity : 1
    # Space Complexity: 0 
    def test(self, testdata, k = None, dist_method = None, probs = False):
        """Member Function; Overridden from SuperClass; tests the classifier on the imported data
 
        
        Paramters
        ~~~~~~~~~
        testdata [numpy.ndarray]: A 2D array of numeric values with shape (m,n)
        k [int]: A postive integer dictating the number of neighbors: Default: 10
        dist_method [str]: the distance metric to use; must be 'euclidean' or 'manhattan'; Default: 'euclidean'
        probs [boolean]; Whether the returned predictions should be a probability
        
        Returns
        ~~~~~~~
        self._preds [numpy.ndarray]: an array of predictions for the test data provided
        
        """
        
        # for complexity analysis:
            # self._data has shape (n,m)
            # testdata has shape (j,m)
            # k represents the number of neighbors (k)
            # l represents the number of unique labels in self._labels
        
        # if _data or _labels arent defined, train() has been run yet. raise error
        # Time Complexity : 5 (2 get__attr__ methods and 3 boolean comparisons)
        # Space Complexity: 0
        if not isinstance(self._data, np.ndarray) or not isinstance(self._labels, np.ndarray):
            raise NotImplementedError(
                'Train Data and Labels have not been initialized; execute train() method first'
            )
            
        # if testdata isn't an numpy.ndarray raise error
        # Time Complexity : 2 (1 get__attr__ method, 1 boolean comparison)
        # Space Complexity: 0
        if not isinstance(testdata, np.ndarray):
            raise TypeError(
                    'Test Data must be type numpy.ndarray with shape (j,m); got ' + str(type(testdata))
                )
            
        # if testdata doesn't match column shape of self,_data, raise error
        # Time Complexity : 5 (2 get__attr__ methods, 2 slices, 1 boolean comparison)
        # Space Complexity: 0            
        if testdata.shape[1] != self._data.shape[1]:
            raise ValueError(
                'Dimension Mismatch: Test Data and Train Data must be numpy.ndarray with shapes (j,m) and (n,m); got ' + str(testdata.shape) + 'and ' + str(self._data.shape)
            )
            
        # If k is not provided, use default from constructor
        # Time Complexity : 2 (1 get__attr__ method, 1 boolean comparison)
        # Space Complexity: 0 (k already allocated)  
        if k == None:
            k = self._k
        
        # if k isn't a positive integer
        # Time Complexity : 4 (1 get__attr__ methods, 3 boolean comparison)
        # Space Complexity: 0   
        if  not isinstance(k, int) or k < 1:
            raise ValueError(
                'k must be a positive integer'
            )
            
        # If dist_method is not provided, use default from constructor
        # Time Complexity : 2 (1 get__attr__ method, 1 boolean comparison)
        # Space Complexity: 0 (dist_method already allocated)  
        if dist_method == None:
            dist_method = self._dist_method
        
        # if dist_method isnt defined
        # Time Complexity : 3 (3 boolean comparison)
        # Space Complexity: 0  
        if dist_method not in ['euclidean', 'manhattan']:
            raise ValueError(
                "distance must be 'euclidean' or 'manhattan'; got " + str(distance)
            )
        
         # if probs is true
        if probs:
            # initialze preds as a jxl array
            self._probs = np.zeros(self._uniquelabels*len(testdata)).reshape(len(testdata),self._uniquelabels)
        
        else:
            # inialize _preds as an empty numpy array
            # Time Complexity : j (j iterations)
            # Space Complexity: j (create array of length j)  
            self._preds = np.array([0 for i in range(len(testdata))])
        
        # inialize _preds as an empty numpy array
        # Time Complexity : 2j (testrow updated j times, j boolean comparisons)
        # Space Complexity: 1 (initialization of testrow)          
        for testrow in range(len(testdata)):
            
            # inialize a 2D array 
            # Time Complexity : j(k + 1) (initialize k arrays and concat into 2D array j times)
            # Space Complexity: 2k (initialization 2D array of shape (k,2))             
            knn = np.array(
                [np.array([np.inf, np.inf]) for i in range(k)]
            )
            
            # inialize _preds as an empty numpy array
            # Time Complexity : 2jn (testrow updated jn times, jn boolean comparisons)
            # Space Complexity: 1 (initialization of testrow)               
            for trainrow in range(len(self._data)):
                
                
                # if method is 'euclidean'
                # Time Complexity : jn (jn boolean comparisons)
                # Space Complexity: 0   
                if dist_method == 'euclidean':
                    
                    # compute euclidean distance
                    # Time Complexity : 3mjn (m substraction operations, m square operations, summation of m elements, jn times)
                    # Space Complexity: 1 (dist initialized, computations immediately freed from memory)   
                    dist = sum(
                        [
                            (self._data[trainrow,i] - testdata[testrow,i])**2 for i in range(testdata.shape[1])
                        ]
                    )
                
                # otherwise, method is 'manhattan'
                else:
                    # compute manhatten distance
                    # Not computed for default case
                    dist = sum(
                        [
                            abs(self._data[trainrow,i] - testdata[testrow,i]) for i in range(testdata.shape[1])
                        ]
                    )
                
                # if dist is less than the largest of the k smallest distances
                # Time Complexity : 2jn (jn slices, jn boolean comparisons)
                # Space Complexity: 0
                if dist < knn[-1,1]:
                    
                    # Initialize indice that dist will place in
                    # Best Time Complexity: jk (distances are in sorted in ascending order of iteration, first k distances are sorted in descending order)
                    # Worst Time Complexity: jn (distances after the first k distances are in descending order and
                    # all fall between the largest and second largest of the k smallest distances)
                    # Space Complexity : 1 (knn_ind initialized)
                    knn_ind = 0
                    
                    # While the value held at the knn_ind indice is less than dist
                    # Best Time Complexity: jk (distances are in sorted in ascending order of iteration, first k distances are sorted in descending order)
                    # Worst Time Complexity: jkn (distances after the first k distances are in descending order and
                    # all fall between the largest and second largest of the k smallest distances)
                    # Space Complexity : 0 
                    while knn[knn_ind,1] < dist:

                        # increment knn_ind indice is less than dist
                        # Best Time Complexity: 0 (distances are in sorted in ascending order of iteration, first k distances are sorted in descending order)
                        # Worst Time Complexity: jkn (distances after the first k distances are in descending order and
                        # all fall between the largest and second largest of the k smallest distances)
                        # Space Complexity : 0 (already initialized)
                        knn_ind += 1
                    
                    # Insert dist and its indice if self._data into array, remove last element
                    # Best Time Complexity: 4jk (distances are in sorted in ascending order of iteration, first k distances are sorted in descending order)
                    # Worst Time Complexity: 4jn (distances after the first k distances are in descending order and
                    # all fall between the largest and second largest of the k smallest distances)
                    # Space Complexity : 0 (one element added and one removed)
                    knn = np.vstack(
                        [
                            knn[:knn_ind, :],
                            np.array(
                                [trainrow,dist]
                            ),
                            knn[knn_ind: -1, :]
                        ]
                    )
    
            # get the value counts of each label indexed by elements of knn array
            # Time Complexity: 2jk (k checks if in unique list, k counter increments, j times)
            # Best Space Complexity : 2 (1 label, 1 count) (all labels returned are the same)
            # Worst Space Complexity : 2l (l labels, l counts) (each unique label is represented)
            val_counts = np.unique(
                self._labels[knn[:,0].astype(int)],
                return_counts = True
            )
            
            
            # if probs is true
            if probs:
                
                # get the sum of classes
                total = sum(val_counts[1])
                
                # for each class in valcounts
                for _,i in enumerate(val_counts[0]):
                    
  
                    # add the probability of the class to the probs array
                    self._probs[testrow, int(i)] = val_counts[1][_]/total
            
            
            else:
                # update the label the testrow indice of preds array with the mode of the returned labels
                # Best Time Complexity: 2j (j indices accessed, j traversal of length 1 array)
                # Worst Time Complexity: 2jl (j indices accessed, j traversal of length l array)
                # Space Complexity : 1 (max_ind takes up one space, one element updated)
                max_ind = [i for i in val_counts[1]].index(max(val_counts[1]))
                self._preds[testrow] = val_counts[0][max_ind]
         
        # return self._preds
        # Time Complexity: 1
        # Space Complexity: 0
        if probs:
            return self._probs
        else:
            return self._preds
    
    
        ##############################
        # Complexity Analysis
        #############################
        
        
        # for reference:
            # self._data has shape (n,m)
            # testdata has shape (j,m)
            # k represents the number of neighbors (k)
            # l represents the number of unique labels in self._labels
        
        # Worst Case: 
            # distances after the first k distances are in descending order and 
            # all fall between the largest and second largest of the k smallest distances,
            # and the labels of the training data are represented with indeices of the k smallest distances
            
            # for fixed m,n,l,k, and variable j (number of rows in test data)
            # Time Complexity : T(j) = 3mjn + 2jkn + 10jn + 3jk + 2jl + 4j + 25
            
            # Let g(j) = j
            # let c = 3mn + 2kn + 10n + 3k + 2l + 5
            # Let j0 = 25
            # Then T(j) <= c*g(j) for all j >= j0 
            # Therefore T(j) has linear Big-Oh given that m, n, l, and k are fixed
            
            # for fixed l,k, and variable j (number of rows in test data)
            # Space Complexity : S(j) = j + 2l + 2k + 4
            
            # Let g(j) = j
            # let c = 2
            # Let j0 = 2l + 2k + 4
            # Then S(j) <= c*g(j) for all j >= j0 
            # Therefore S(j) has linear Big - Oh given that l and k are fixed



    

class Tree:
    """
    Abstract Base Class Tree Structure; Contains nested Node Class
    
    
    Attributes:
    ~~~~~~~~~~~~~~~~~~~
    root [Node]; the root of the tree, holds a Node object; Default: None
    

        
    """
    
    class Node:
        
        """
        Abstract Base Class Tree Node


        Attributes:
        ~~~~~~~~~~~~~~~~~~~
        _leftChild [Node]; the left child of the current node, holds a Node object; Default: None
        _rightChild [Node]; the right child of the current node, holds a Node object; Default: None


        """
        
        # initialize default arguments
        _leftChild = None 
        _rightChild = None  
        
        def __init__(self, value):
            """
            Contructor for the Node Class;


            Parameters
            ~~~~~~~~~~~~~
            value [object]: The information stored in the node; can be any object
            
            
            Returns
            ~~~~~~~~~~~~
            [Node]: a Node Class object
            
            """
            
            #assign value to node
            self.value = value

        
                
        def hasChildren(self):
            """
            Base Method for Node Class; returns whether the node has children
            
            
            Parameters
            ~~~~~~~~~~~~~
            None
            
            
            Returns
            ~~~~~~~~~~~~
            [boolean]: whether the node has children
                        
            """
            
            return self._leftChild != None and self._leftChild != None
    
    
    def __init__(self):
        """
        Contructor for the Tree Class; initializes root to None
        
        """
        self.root = None

    
    def __str__(self):
        """
        print() method for Tree class
        

        Parameters
        ~~~~~~~~~~~~~
        None


        Returns
        ~~~~~~~~~~~~
        [string]: a string representation of the Decision Tree
        
        """
        # call recursive toString method
        return self._toString(self.root)

    # recursive depth first search to create syntree formatted string
    def _toString(self, node):

        """
        Recursive DFS method to create formatted string
        
        Parameters
        ~~~~~~~~~~~~~~
        node [Node]: a Node Class Object
        
        
        Returns
        ~~~~~~~~~~~~~~
        [string]: a string representation of the Decision Tree
        
        """
        # if node is none, return empty string
        if node == None:
            return ''

        # call _toString on left and right child nodes
        lhs = self.__toString(node.leftChild)
        rhs = self.__toString(node.rightChild)

        # return the nodes payload, and the left and rhs strings, wrapped in brackets
        return '[' + str(node.value) + ' ' + lhs + rhs + ']'

    
    
    
class kdTreeKNNClassifier(ClassifierAlgorithm, Tree):
    """Subclass of ClassifierAlgorithm Abstract Base Class and Tree Abstract Base Class;
    performs KNN algorithm on a k-dimensional tree
 
    """
    

        
    class kdNode(Tree.Node):
        """
        Node Subclass Inherited from Tree.Node Abstract Base Class; creates a kdTree node


        Inherits
        ~~~~~~~~~~
        _leftChild
        _rightChild
        hasChildren()



        """
        
        
        class HeapSort:
            """
            A Heap Sorting Algorithm Class, used to sort a 2 dimensional array by a selcted column index


            """
            def __init__(self, lst, on = 0):
                """
                Constructor for HeapSort class

                Paramters
                ~~~~~~~~~
                lst [np.ndarray]: A multidimensional array
                on [int]; The column index to sort on


                Returns
                ~~~~~~~
                [HeapSort]: a Heapsort Class Object 

                """

                # set the column to sort on
                self._on = on

                # heapify the list
                self.heapify(lst)



            def heapify(self, lst):
                """
                Transform lst to follow heap properties

                Paramters
                ~~~~~~~~~
                lst [np.ndarray]: A multidimensional array


                Returns
                ~~~~~~~
                None             


                """

                # only need to heap down on half the elements
                i = len(lst) // 2

                # deep copy of lst
                self.heapList = [ele for ele in lst]

                # set current size
                self.currentSize = len(lst)

                # iteratively heapdown
                while (i >= 0):
                    self.heapDown(i)
                    i -= 1


            def sort(self):
                """
                Iteratively unheap the Heap, returning a sorted descending list


                Paramters
                ~~~~~~~~~
                None


                Returns
                ~~~~~~~
                [np.ndarray]: The sorted array in descending order

                """

                # pop each value off heap, return sorted list
                return np.array([self._pop() for i in range(self.currentSize)])



            def heapDown(self, i):
                """
                Performs a heap down operation on a node

                Paramters
                ~~~~~~~~~
                i [int]: index of the heap


                Returns
                ~~~~~~~
                None           

                """

                # while a child of i exists
                while (2 * i + 1 ) < self.currentSize:

                    # get the index of the minimum child
                    mc = self.maxChild(i)

                    # if parent is less than the child
                    if self.heapList[i][self._on] < self.heapList[mc][self._on]:

                        # swap the parent and child
                        self.heapList[i], self.heapList[mc] = self.heapList[mc], self.heapList[i]

                    # update i to the minimum child index
                    i = mc

            def maxChild(self, i):
                """
                finds the index of the of the maximum child of node i

                Paramters
                ~~~~~~~~~
                i [int]: index of the heap


                Returns
                ~~~~~~~
                maxidx [int]: the index of the of the maximum child of node i         

                """

                # assume left child is min
                maxidx = 2 * i + 1
                maxchild = self.heapList[maxidx][self._on]

                # if right child exists
                if maxidx + 1 < self.currentSize:

                    # if right child is greater than leftchild
                    if self.heapList[maxidx + 1][self._on] > maxchild:

                        # min child index is right child
                        maxidx += 1

                # return index of minimum child
                return maxidx

            def _pop(self):
                """
                removes the top of the heap and returns it

                Paramters
                ~~~~~~~~~
                None


                Returns
                ~~~~~~~
                retval [np.ndarray]: the object popped from the top of the heap


                """

                # get the value that is going to be popped off
                retval = self.heapList[0]

                # replace the top of the heap with the bottom of the heap
                self.heapList[0] = self.heapList[self.currentSize - 1]

                # decrement current size
                self.currentSize -= 1

                # remove the last element in list (now duplicated to be in first index)
                self.heapList.pop() 

                # heap down on the first index
                self.heapDown(0)

                # return the popped value
                return retval
        
        
        # set default parameters
        # Time Complexity: 4
        # Space Complexity: 4
        _splitIndex = None
        _splitCondition = None
        _splitFeature = None
        _prediction = None

        
        # Call Constructor
        # Time Complexity: 1
        # Space Complexity: 1
        def __init__(self, data, indices, labels):
            """
            Overridden Constructor for Node Subclass


            Attributes
            ~~~~~~~~~~~~~~~

            _data [np.ndarray]: a reference pointer to a np.ndarray; training data
            _indices[list[int]]: a list of row indices that define the subset of _data and _labels for which the node
            _labels [np.ndarray]: a refrence pointer to a np.ndarray; training labels
            
            


            """

            # initialize attributes
            # Time Complexity:  3
            # Space Complexity: 2 + i (1 float, 2 pointer references, list of length i)
            self._indices = indices
            self._labels = labels
            self._data = data

            
        # function call
        # Time Complexity : 1
        # Space Complexity: 0
        def isPure(self):
            """
            returns whether the node is pure or not (only has one point defined on it)
            
            
            Parameters
            ~~~~~~~~~~~~
            None
            
            
            Returns
            ~~~~~~~~~~~~
            [boolean]: Whether or not the node object is pure

            """
            # Time Complexity: 3 (len(), boolean comparison, return)
            # Space Complexity: 0
            return len(self._indices) == 1
        
        
        
        def hasLeftChild(self):
            """
            returns whether the node has a left child
            
            
            Parameters
            ~~~~~~~~~~~~
            None
            
            
            Returns
            ~~~~~~~~~~~~
            [boolean]: Whether or not the node has a left child

            """
            
            return self._leftChild != None
        
        
        def hasRightChild(self):
            """
            returns whether the node has a right child
            
            
            Parameters
            ~~~~~~~~~~~~
            None
            
            
            Returns
            ~~~~~~~~~~~~
            [boolean]: Whether or not the node has a right child

            """
            return self._rightChild != None
        
        
        
        # function call
        # Time Complexity: 1
        # Space Complexity: 0
        def findBestSplit(self, axis):
            """
            Subclass method that finds the best split on the node and returns information about it
            
            
            Parameters
            ~~~~~~~~~~~~~
            axis [int]: The column index that the split occurs on
            
        
            Returns
            ~~~~~~~~~~~~~
            leftIndices [list[int]]: a list of row indices that are defined in left child node
            rightIndices [list[int]]: a list of row indices that are defined in right child node

            """
            
            # set the split feature axis column
            # Time Complexity: 1
            # Space Complexity: 0 (already initialized)
            self._splitFeature = axis
            
            
            # concatenate the subsetted data and the indices
            # perform a heapsort, sorting along the axis column
            # Time Complexity: n log n + 4 (heapsort, 1 slice, 1 concat, 1 transpose, 1 assignment)
            # Space Complexity 2n (n rows, 2 columns)
            sortedData = self.HeapSort(
                np.vstack(
                    [
                        self._data[self._indices, axis],
                        self._indices
                    ]
                ).transpose(),
                on = 0
            ).sort()
            
            
            # get the middle index of the sorted data
            # Time Complexity: 2 (len(), floor division)
            # Space Complexity: 1
            medIdx = len(sortedData)//2
            
            # set the splitting condition of the node
            # Time Complexity: 2 (1 slice, 1 assignment)
            # Space Complexity: 0 (Already initialized)
            self._splitCondition = sortedData[medIdx,0]
            
            
            
            # get the left and right indices
            # Time Complexity: 4n + 4 (2 slices, n boolean operations, n casts as int, 2 times)
            # Space Complexity: n (n rows partitioned into two arrays)
            leftIndices = sortedData[sortedData[:,0] > self._splitCondition, 1].astype(int)
            rightIndices = sortedData[sortedData[:,0] <= self._splitCondition, 1].astype(int)
            
            
            
            # set the split row index of node
            # Time Complexity: 2 (1 slice , 1 assignment)
            # Space Complexity: 0 (Already initialized)
            self._splitIndex = rightIndices[0]
            
            # set the prediction label of node
            # Time Complexity: 2 (1 slice , 1 assignment)
            # Space Complexity: 0 (Already initialized)
            self._prediction = self._labels[self._splitIndex]
            
            
            # remove the first point in the rightIndices (point is saved in splitIndex attribute)
            # Time Complexity: 2 (1 slice, 1 assingment)
            # Space Complexity: -1 (1 space is freed after garbage collection)
            rightIndices = rightIndices[1:]
            
            
            # return the left and right indices
            # Time Complexity: 1
            # Space Complexity: 0
            return leftIndices, rightIndices
            
            
        
        # function call
        # Time Complexity: 1
        # Space Complexity: 0
        def _goLeft(self, data):
            """
            Subclass method that determines whether the test algorithm should traverse left

            Parameters
            ~~~~~~~~~~~~~~
            data [np.ndarray] : a 1xm test array
            
            
            Returns
            ~~~~~~~~~~~~~~
            [boolean]: Whether or not the test algorithm should traverse left
            
            
            """
            # return boolean
            # Time Complexity: 3 (1 slice, 1 boolean, 1 return)
            # Space Complexity: 0
            return data[self._splitFeature] > self._splitCondition

        
    def __init__(self, k = 10):
        """
        Constructor for Decision Tree Subclass


        Parameters
        ~~~~~~~~~~~~~~~
        k [int]: the number of neighbors for the KNN algorithm 
        


        Returns
        ~~~~~~~~~~~~~~~
        [kdTreeKNNClassifier]: a kdTreeKNNClassifier object
        
        """
        
        # if k isn't a positive integer, raise error
        if  not isinstance(k, int) or k < 1:
            raise ValueError(
                'k must be a positive integer'
            )
        
        # explicitly call superconstuctors
        ClassifierAlgorithm.__init__(self)
        Tree.__init__(self)
        
        # set hyperparameters
        self.k = k
    
    
    # function call
    # Time Complexity: 1
    # Space Complexity: 0
    def train(self, data, labels, k = None):
        """
        Train method that builds Decision Tree nodes
        
        Parameters
        ~~~~~~~~~~~~~~
        data [np.ndarray]: a nxm array used for training
        labels [np.ndarray]: a 1xn array representing the class labels of the training data
        k [int]: the number of neighbors for the KNN algorithm 
        
        Returns
        ~~~~~~~~~~~~~~
        [None]
        
        """
        
        # create label dictionary for probability mapping
        # Time complexity: 2L
        # Space Complexity: 2L
        self._labelDict = {lab: ind for ind, lab in enumerate(np.unique(labels))}
        
        
        # Assign data and labels to predefined attributes
        # Time Complexity: 2 
        # Space Complexity: nm + n
        self._data = data
        self._labels = labels
        
        
        # Get the number of unique labels for calculating probability
        # Time Complexity: 3 (len(), set(), 1 assignment)
        # Space Complexity: 1
        self._uniquelabels = len(set(self._labels))
        
        
        # if k not provided
        # Time Complexity: 1
        # Space Complexity: 0
        if k == None:
            
            # assign k from attribute value
            # Time Complexity: 1
            # Space Complexity: 0
            k = self.k
        
        
        # set the dimensions of the columns
        # Time Complexity: 2 (1 slice, 1 assignment)
        # Space Complexity: 
        self._dims = data.shape[1]
        
        # create root node
        # Time Complexity: 8 (__init__)
        # Space Complexity: n + 7  (__init__)
        self.root = self.kdNode(data, np.arange(len(data)), labels)
        
        
        # Recursively build tree
        # Time Complexity: n
        # Space Complexity: n
        self._trainRecursive(self.root, 0)

        
    
    
    # function call
    # Time Complexity: 1
    # Space Compelxity: 0
    def _trainRecursive(self, node, depth):
        """
        Recursive train method that builds Decision Tree nodes
        
        Parameters
        ~~~~~~~~~~~~~~
        node [kdNode]: a kdNode subclass
        depth [int]: the current depth of the tree at current node
        
        Returns
        ~~~~~~~~~~~~~~
        [None]
        
        """
        
        
        
        # check if node is pure (exit condition)
        # Time Complexity: 3n (isPure, n times)
        # Space Complexity: 0
        if node.isPure():
            
            
            # set the private attributes for prediction and testing
            # Time Complexity: 8((n+1)/2) (4 assignments, 3 slices, 1 modulo, (n+1)/2 leaf nodes)
            # Space Complexity: 0 (already initialized)
            node._splitIndex = node._indices[0]
            node._prediction = node._labels[node._splitIndex]
            node._splitFeature = depth%self._dims
            node._splitCondition = node._data[node._splitIndex, node._splitFeature]

        
            # return
            # Time Complexity: (n+1)/2 (return (n+1)/2 times)
            # Space Complexity: 0
            return
        
        
        # set the axis column to split on
        # Time Complexity: 2((n-1)/2) ((n-1)/2 modulos, (n-1)/2 assignments)
        # Space Complexity: 1
        axis = depth%self._dims
       
        # get the indices of the best split on the axis
        # Time Complexity: ((n-1)/2)(nlogn + 4n + 20) (findBestSplit(), (n-1)/2) times)
        # Space Complexity: 4n + 1 (findbestsplit, n total spaces to fit left and right indices)
        leftIndices, rightIndices = node.findBestSplit(axis)
        

        # if leftIndices is not empty
        # Time Complexity: 2((n-1)/2) ((n-1)/2) len(), (n-1)/2) boolean)
        # Space Complexity: 0
        if len(leftIndices) > 0:
            
            # initialize left child
            # Time Complexity accounts for right child as well
            # Time Complexity: 8n (__init__, n times)
            # Space Complexity: n(n + 7) (__init__, n times)
            node._leftChild = self.kdNode(node._data, leftIndices, node._labels)
            
            # recursively train on left child
            # Complexity already accounted for
            self._trainRecursive(node._leftChild, depth+1)
        
        # if rightIndices is not empty
        # Time Complexity: 2((n-1)/2) ((n-1)/2) len(), (n-1)/2) boolean)
        # Space Complexity: 0
        if len(rightIndices) > 0:
            
            # initialize right child
            # Complexity already accounted for
            node._rightChild = self.kdNode(node._data, rightIndices, node._labels)
            
            # recursively train on right child
            # Complexity already accounted for
            self._trainRecursive(node._rightChild, depth+1)
        
        
            
            
            
                
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    
    # Complexity Analysis
    
    # For reference:
        # n = number of rows
    
    
    # for variable n (number of rows in data)
        # Time Complexity : T(n) = ((n^2 -n)/2)logn + 2n^2 + 28n -8

        # Let g(n) = n^2 log n
        # let c = 10
        # Let n0 = 6
        # Then T(n) <= c*g(n) for all n >= n0 
        # Therefore T(n) has n^2 log n Big-Oh

    # for variable n (number of rows in data)
        # Space Complexity : S(n) = n^2 + 11n +2

        # Let g(n) = n^2
        # let c = 3
        # Let n0 = 6
        # Then S(n) <= c*g(n) for all n >= n0 
        # Therefore S(n) has n^2 Big - Oh         
            
     
    # Comparison to simpleKNNClassifier
        # The kdTreeKNNClassifier has a training Big-Oh of n^2logn
        # The simpleKNNClassifier has a constant training Big-Oh because the data is only assigned in the object
        #
        # These training methods aren't very comparable even though they are steps in a similar process
        # The kdTree method attempts to pre-organize the data in such a manner that testing time will be quicker
        # Ultimately, this does make the training algorithm for the kdTreeKNNClassifier less efficient
            
            
    # function call
    # Time Complexity: 1
    # Space Complexity: 0
    def test(self, testdata, k = None, probs = False):        
        """
        Recursive test method that predicts the class of test data
        
        Parameters
        ~~~~~~~~~~~~~~
        data [np.ndarray] : a jxm test array
        probs [boolean]; Whether the returned predictions should be a probability
        
        Returns
        ~~~~~~~~~~~~~~
        [np.ndarray]: an array of class predictions provided test data
        
        """
        
        # If k is not provided, use default from constructor
        # Time Complexity : 2 (1 get__attr__ method, 1 boolean comparison)
        # Space Complexity: 0 (k already allocated)  
        if k == None:
            k = self.k
        
        
        # if k isn't a positive integer
        # Time Complexity : 4 (1 get__attr__ methods, 3 boolean comparison)
        # Space Complexity: 0   
        if  not isinstance(k, int) or k < 1:
            raise ValueError(
                'k must be a positive integer'
            )
            
        # if probs is true
        # Time Complexity: 1
        # Space Complexity: 0
        if probs:
            # initialze preds as a jxl array
            self._probs = np.zeros(self._uniquelabels*len(testdata)).reshape(len(testdata),self._uniquelabels)
        
        # otherwise, generate class predictions
        else:
            # inialize _preds as an empty numpy array
            # Time Complexity : j (j iterations)
            # Space Complexity: j (create array of length j)  
            self._preds = np.array([0 for i in range(len(testdata))])  
        
        
        # for each row in test data
        # Time Complexity: j
        # Space Complexity: 1 (testrow)
        for testrow in range(len(testdata)):
            
            # Initialize knn matrix
            # Time Complexity: 2jk (2k initialization, j times)
            # Space Complexity: 2k
            knn = np.full((k, 2), np.inf)
            
            # Recursively find KNN
            # Time Complexity: j(2kn + 3mn + 25n - 2) (_testrecusive(), j times)
            # Space Complexity: 4 (_testRecursive)
            knn = self._testRecursive(self.root, testdata[testrow], knn)
            
            # get the value counts of each label indexed by elements of knn array
            # Time Complexity: 2jk (k checks if in unique list, k counter increments, j times)
            # Space Complexity : 2l (l labels, l counts) (each unique label is represented)
            val_counts = np.unique(
                knn[:,0].astype(int),
                return_counts = True
            )
            

            
            # if probs is true
            # ignored for time complexity
            if probs:
                
                # get the sum of classes
                total = sum(val_counts[1])
                
                # for each class in valcounts
                for _,i in enumerate(val_counts[0]):
                    
  
                    # add the probability of the class to the probs array
                    self._probs[testrow, int(i)] = val_counts[1][_]/total
            
            
            else:
                # update the label the testrow indice of preds array with the mode of the returned labels
                # Time Complexity: 2jl (j indices accessed, j traversal of length l array)
                # Space Complexity : 1 (max_ind takes up one space, one element updated)
                max_ind = [i for i in val_counts[1]].index(max(val_counts[1]))
                self._preds[testrow] = val_counts[0][max_ind]
         
        
        # if probs, returnt the probabilities
        # Time complexity: 1
        # Space Complexity: 0
        if probs:
            return self._probs
        
        # otherwise, return the predictions
        else:
            # return self._preds
            # Time Complexity: 1
            # Space Complexity: 0
            return self._preds
        
        
            
            
    
    # function call
    # Time Complexity: n
    # Space Complexity: 0
    def _testRecursive(self, node, data, knn):
        """
        Recursive test method that predicts the class of a test point
        
        Parameters
        ~~~~~~~~~~~~~~
        node [QuantNode/QualNode]: a Node subclass
        data [np.ndarray] : a 1xm test array
        knn [np.ndarray]:  
        
        Returns
        ~~~~~~~~~~~~~~
        knn [np.ndarray]: the class prediction of the test point
        
        """
        
        # if node is undefined, return knn
        # Time Complexity: 3n - 2 (n boolean, 2(n-1) boolean for empty children of leaf nodes)
        # Space Complexity: 0
        if node == None:
            # Time Complexity: 2n - 2 (2(n-1) returns for empty children of leaf nodes)
            # Space Complexity: 0
            return knn



        # compute euclidean distance
        # Time Complexity : 3mn (m substraction operations, m square operations, summation of m elements, n times)
        # Space Complexity: 1 (dist initialized, computations immediately freed from memory)   
        dist = sum(
            [
                (self._data[node._splitIndex,i] - data[i])**2 for i in range(len(data))
            ]
        )
        

        # if dist is less than the largest of the k smallest distances
        # Time Complexity : 2n (n slices, n boolean comparisons)
        # Space Complexity: 0
        if dist < knn[-1,1]:

            # Initialize indice that dist will place in
            # Time Complexity: n (distances after the first k distances are in descending order and
            # all fall between the largest and second largest of the k smallest distances)
            # Space Complexity : 1 (knn_ind initialized)
            knn_ind = 0

            # While the value held at the knn_ind indice is less than dist
            # Time Complexity: kn (distances after the first k distances are in descending order and
            # all fall between the largest and second largest of the k smallest distances)
            # Space Complexity : 0 
            while knn[knn_ind,1] < dist:

                # increment knn_ind indice is less than dist
                # Time Complexity: kn (distances after the first k distances are in descending order and
                # all fall between the largest and second largest of the k smallest distances)
                # Space Complexity : 0 (already initialized)
                knn_ind += 1
                
        
            # Insert dist and its indice if self._data into array, remove last element
            # Time Complexity: 4n (distances after the first k distances are in descending order and
            # all fall between the largest and second largest of the k smallest distances)
            # Space Complexity : 0 (one element added and one removed)    
            knn = np.vstack(
                [
                    knn[:knn_ind, :],
                    np.array(
                        [node._labels[node._splitIndex],dist]
                    ),
                    knn[knn_ind: -1, :]
                ]
            )
        
        # update pathNode and otherNode according to the correct path
        # Time Compelexity: 6n (goLeft(), 2 assingments, n times)
        # Space Complexity: 2
        if node._goLeft(data):
            pathNode = node._leftChild
            otherNode = node._rightChild
            
        else:
            pathNode = node._rightChild
            otherNode = node._leftChild
            
        
        # traverse path node
        # Complexity already considered
        knn = self._testRecursive(pathNode, data, knn)
        
        
        # if radial distance to linear boundary is less than at least one of the minimum distances
        # Time Complexity: 5n (2 slices, 1 subtraction, 1 abs(), 1 boolean, n times)
        # Space Compelexity: 0
        if abs(data[node._splitFeature] - node._splitCondition) < knn[-1,1]:
            
            # Traverse the other child node
            # Complexity already considered
            knn = self._testRecursive(otherNode, data, knn)         
        
        # return knn
        # Time Complexity: n
        # Space Complexity: 0
        return knn               
        
        
        ##############################
        # Complexity Analysis
        #############################
        
        
        # for reference:
            # self._data has shape (n,m)
            # testdata has shape (j,m)
            # k represents the number of neighbors (k)
            # l represents the number of unique labels in self._labels
        
        # Worst Case: 
            # All radial distances to the linear boundaries are less then at least one of the k minimum values 
            # The algorithm cannot prune the tree and must traverse all n nodes of the kdTree
            
        # for fixed m,n,l,k, and variable j (number of rows in test data)
            # Time Complexity : T(j) = 2jkn + 3jmn + 4jk + 2jl + 25jn + 10
            
            # Let g(j) = j
            # let c = 2kn + 3mn + 4k + 2l + 25n + 1
            # Let j0 = 10
            # Then T(j) <= c*g(j) for all j >= j0 
            # Therefore T(j) has linear Big-Oh given that m, n, l, and k are fixed
            
            # for fixed l and variable j (number of rows in test data)
            # Space Complexity : S(j) = j + 4L + 6
            
            # Let g(j) = j
            # let c = 2
            # Let j0 = 4l + 6
            # Then S(j) <= c*g(j) for all j >= j0 
            # Therefore S(j) has linear Big - Oh given that l is fixed
    
    
    # Comparison to simpleKNNClassifier
        # The kdTreeKNNClassifier has a testing Big-Oh of n
        # The simpleKNNClassifier has a testing Big-Oh of n
        #
        # While these are equal, these are worst case scenarios. Data is often not going to be distributed in such a
        # way that the kdTreeKNNClassifier is unable to prune the tree and be required to visit all nodes. In the 
        # average and more realistic case, the kdTreeKNNClassifier will have a time complexity proportional to log(n)
        # Because we generally want to test more than one point, it is important that the testing time complexity be
        # more efficient. 
        # Ultimately, even though there is a large bottleneck in the training algorithm of the kdTreeKNNClassifier,
        # it is more efficient because it can achieve an average testing time complexity of log(n), compared to the
        # linear time complexity of the brute force search simpleKNNClassifier algorithm.

        
    def _toString(self, node):
        """
        Recursive DFS method to create formatted string
        
        Parameters
        ~~~~~~~~~~~~~~
        node [kdNode]: a kdNode subclass
        
        
        Returns
        ~~~~~~~~~~~~~~
        [string]: a string representation of the Decision Tree
        
        """
        
        # if node is none, return empty string
        if node == None:
            return ''

        # call _toString on left and right child nodes
        lhs = self._toString(node._leftChild)
        rhs = self._toString(node._rightChild)
        
        
        # if node doesnt have a split, nodeVal is 'leaf'
        if node._splitCondition == None:
            nodeVal = 'leaf'
        
        # else if node is a float, nodeVal is '<####'
        elif type(node._splitCondition) == np.float64:
            nodeVal = '<' + '{:.4f}'.format(node._splitCondition)
        
        # otherwise, node is a string
        else:
            nodeVal = node._splitCondition
            
        # return the nodes splitting condition, if any, and the left and rhs strings, wrapped in brackets
        return '[' + nodeVal + ' ' + lhs + rhs + ']'
    

    
    
class DecisionTreeClassifier(ClassifierAlgorithm, Tree):
    """
    Subclass of ClassifierAlgorithm Abstract Base Class and Tree Abstract Base Class; creates a Decision Tree

    """

    class Node(Tree.Node):
        """
        Node Subclass Inherited from Tree.Node Abstract Base Class; creates a Decsion Tree Node


        Inherits
        ~~~~~~~~~~
        _leftChild
        _rightChild
        hasChildren()



        """

        # set default parameters
        # Time Complexity: 2
        # Space Complexity: 2
        _splitCondition = None
        _splitFeature = None

        # Call Constructor
        # Time Complexity: 1
        # Space Complexity: 1

        def __init__(self, data, giniThreshold, indices, labels,
                     val_counts=None, prediction=None, gini=None
                    ):
            """
            Overridden Constructor for Node Subclass


            Attributes
            ~~~~~~~~~~~~~~~

            _data [np.ndarray]: a reference pointer to a np.ndarray; training data
            _labels [np.ndarray]: a refrence pointer to a np.ndarray; training labels
            _giniThreshold[float]: the minimum Gini increase required for a split on a node to occur
            _indices[list[int]]: a list of row indices that define the subset of _data and _labels for which the node is defined on
            _val_counts[np.ndarray]: a np.ndarray created from using the np.unique(return_counts = True) method; Computed if not provided
            _prediction[int/float/string]: The class prediction for the node; Computed if not provided
            _gini [float]: the Gini Purity of the node; Computed if not provided

            """

            # initialize attributes
            # Time Complexity:  4
            # Space Complexity: 3 + i (1 float, 2 pointer references, list of length i)
            self._giniThreshold = giniThreshold
            self._indices = indices
            self._labels = labels
            self._data = data

            # if val_counts is not provided
            # then none of the other arguments are provided,
            # compute them manually
            # Time Complexity: 1
            # Space Complexity: 0
            if val_counts == None:

                # assign val_counts
                # Time Complexity: 2i + 1 (i checks if value in unique list, i counter updates, 1 slice)
                # Space Complexity : 2l (l labels, l counts)
                self._val_counts = np.unique(
                    self._labels[self._indices],
                    return_counts=True
                )

                # assign prediction
                # Time Complexity: 2l + 4
                # Space Complexity : 1
                self._prediction = self._computePrediction(self._val_counts)

                # assign gini
                # Time Complexity: 5l + 2
                # Space Complexity : 1
                self._gini = self._computeGini(self._val_counts)

            # otherwise, attributes are supplied
            else:

                # Time Complexity: 3
                # Space Complexity: 2L + 2
                self._val_counts = val_counts
                self._prediction = prediction
                self._gini = gini

        # function call
        # Time Complexity: 1
        # Space Complexity: 0

        def _computePrediction(self, valCountObj):
            """
            Takes a Value Count Object (from np.unique) and returns the mode
            
            Parameters
            ~~~~~~~~~~~~
            valCountObj [np.ndarray]: np.ndarray created from using the np.unique(return_counts = True) method
            
            
            Returns
            ~~~~~~~~~~~
            [int/string]: the class prediction at that node
            
            """

            # get the index of the mode in the valCountObj
            # Time Complexity: 2l (l indices accessed, traversal of length l array)
            # Space Complexity: 1
            max_ind = [i for i in valCountObj[1]].index(max(valCountObj[1]))

            # return the mode of the valCountObj
            # Time Complexity: 3 (2 slices, 1 return)
            # Space Complexity: 0
            return valCountObj[0][max_ind]

        
        # function call
        # Time Complexity : 1
        # Space Complexity: 0
        def _computeGini(self, valCountObj):
            """
            Takes a Value Count Object (from np.unique) and returns the Gini Impurity

            Parameters
            ~~~~~~~~~~~~
            valCountObj [np.ndarray]: np.ndarray created from using the np.unique(return_counts = True) method
            
            
            Returns
            ~~~~~~~~~~~~
            [float]: The Gini Impurity of the node
            
            """
            # total number of labels
            # Time Complexity: l (iteration of length l to sum value)
            # Space Complexity: 1
            numlabels = sum(valCountObj[1])

            # return the Gini
            # Time Complexity : 4l + 1 (l division, l squares, l updates to i, loop of length l to sum, 1 return)
            # Space Complexity: 0
            return 1 - sum([(i/numlabels)**2 for i in valCountObj[1]])

        # function call
        # Time Complexity : 1
        # Space Complexity: 0

        def isPure(self):
            """
            returns whether the node is pure or not
            
            
            Parameters
            ~~~~~~~~~~~~
            None
            
            
            Returns
            ~~~~~~~~~~~~
            [boolean]: Whether or not the node object is pure

            """
            # Time Complexity: 2 (boolean comparison, return)
            # Space Complexity: 0
            return self._gini == 0

    class QualNode(Node):
        """
        Subclass for Decision Tree Node Subclass; inherits attributes from Node class; used for qualitative data

        """

        # constructor call
        # Time Complexity: 1
        # Space Complexity: 0
        def __init__(self, data, giniThreshold, indices, labels,
                     val_counts=None, prediction=None, gini=None
                    ):
            """
            Subclass Constructor for Node Subclass


            Parameters
            ~~~~~~~~~~~~~~~

            data [np.ndarray]: a reference pointer to a np.ndarray; training data
            giniThreshold [float]: The minimum improvement in Gini Impurity for a split to occur
            indices[list[int]]: a list of row indices that define the subset of _data and _labels for which the node is defined on
            labels [np.ndarray]: a refrence pointer to a np.ndarray; training labels
            val_counts[np.ndarray]: a np.ndarray created from using the np.unique(return_counts = True) method; Computed if not provided
            prediction[int/float/string]: The class prediction for the node; Computed if not provided
            gini [float]: the Gini Purity of the node; Computed if not provided


            Returns
            ~~~~~~~~~~~~~~~
            [QualNode]: a Node Subclass object
        
            
            """
            # call super constructor
            # Time Complexity: 7l + 2i + 15
            # Space Complexity: 2l + i + 8
            super().__init__(data, giniThreshold, indices, labels,
                     val_counts, prediction, gini
                    )

        # function call
        # Time Complexity: 1
        # Space Complexity: 0

        def findBestSplit(self):
            """
            Subclass method that finds the best split on the node and returns information about it
            
            
            Parameters
            ~~~~~~~~~~~~~
            None
            
        
            Returns
            ~~~~~~~~~~~~~
            minGini [float]: The weighted Gini impurity of the child nodes
            bestSplit [float]: The optimal point to split on
            bestCol [int]: The column index that the split occurs on
            bestLeftIndices [list[int]]: a list of row indices that are defined in left child node
            bestRightIndices [list[int]]: a list of row indices that are defined in right child node
            bestLeftValCounts [np.ndarray]: np.ndarray created from using the np.unique(return_counts = True) method
            bestRightValCounts [np.ndarray]: np.ndarray created from using the np.unique(return_counts = True) method
            bestLeftGini [float]: the Gini Impurity of the left child
            bestRightGini [float]: the Gini Impurity of the left child


            """

            # initialize minimum gini
            # Time Complexity: 1
            # Space Complexity : 1
            minGini = np.inf

            # for column in the training data
            # Time Complexity: m (m updates to column)
            # Space Compelxity: 1 (column)
            for column in range(self._data.shape[1]):

                # subset data by the data defined by the node
                # Time Complexity: m (1 slice, m times)
                # Space Complexity: mi (i elements in indices, m times)
                workingData = self._data[self._indices, column]

                # get the unique elements in workingData
                # Time Complexity: mi (loop thorugh WorkingData, m times)
                # Space Complexity: l
                qualities = np.unique(workingData)

                # for each unique quality in column
                # Time Complexity: qm (q updates to qual, m times)
                # Space Compelxity: 1 (qual)
                for quals in qualities:

                    # Get left and right indices
                    # Time Complexity: qm(2i + 4) (2i boolean, 2 slices, 2 assignments, qm times)
                    # Space Complexity: i (total indices of both add to i)
                    leftIndices = self._indices[workingData != quals]
                    rightIndices = self._indices[workingData == quals]

                    # calculate leftValCounts and rightValCounts
                    # Time Complexity: qm(4i + 4) (2i total checks if value in unique list, 2i total counter updates, 4 slice, qm times)
                    # Space Complexity : 4l (2l labels, 2l counts)
                    leftValCounts = np.unique(
                        self._labels[leftIndices],
                        return_counts=True
                    )

                    rightValCounts = np.unique(
                        self._labels[rightIndices],
                        return_counts=True
                    )

                    # calculaute left and right gini
                    # Time Complexity: qm(2(5L +2) + 2) (2 computeGini, 2 assingment, qm times)
                    # Space Complexity: 2
                    leftGini = self._computeGini(leftValCounts)
                    rightGini = self._computeGini(rightValCounts)

                    
                    # get weights of each child (ensures more equal splits)
                    # Time Complexity: 8qm (4 len(), 2 division, 2 assingment)
                    leftWeight = len(leftIndices)/len(self._indices)
                    rightWeight = len(rightIndices)/len(self._indices)
                    
                    
                    # calculaute mean Gini
                    # Time Complexity: 4qm (1 sum, 2 multiplcation, 1 assignment, qm times)
                    # Space Complexity: 1
                    newGini = leftWeight*leftGini + rightWeight*rightGini
                    
                    
                    
                    # If newGini is better than the current gini
                    # Time Compelxity: qm (qm boolean)
                    # Space Complexity: 0
                    if newGini < minGini:

                        # WORST CASE: qm times

                        # update maxGini, bestSplit, and best col
                        # Time Complexity: 3qm
                        # Space Complexity: 2 (maxGini already initialized)
                        minGini = newGini
                        bestSplit = quals
                        bestCol = column

                        # update left and right best indices
                        # Time Complexity: 2qm
                        # Space Complexity: i (total i indices to partition)
                        bestLeftIndices = leftIndices
                        bestRightIndices = rightIndices

                        # update left and right best valcounts
                        # Time Complexity: 2qm
                        # Space Complexity: 4l (2l labels, 2l counts)
                        bestLeftValCounts = leftValCounts
                        bestRightValCounts = rightValCounts

                        # update best left and right gini
                        # Time Complexity: 2qm
                        # Space Complexity: 2
                        bestLeftGini = leftGini
                        bestRightGini = rightGini

                        
            minGini
            # return values
            # Time Complexity: 1
            # Space Complexity: 0
            return minGini, bestSplit, bestCol, bestLeftIndices, bestRightIndices, bestLeftValCounts, bestRightValCounts, bestLeftGini, bestRightGini

        # function call
        # Time Complexity: 1
        # Space Complexity: 0

        def _goLeft(self, data):
            """
            Subclass method that determines whether the test algorithm should traverse left

            Parameters
            ~~~~~~~~~~~~~~
            data [np.ndarray] : a 1xm test array
            
            
            Returns
            ~~~~~~~~~~~~~~
            [boolean]: Whether or not the test algorithm should traverse left
            
            
            """
            
            # return boolean
            # Time Complexity: 3 (1 slice, 1 boolean, 1 return)
            # Space Complexity: 0
            return data[self._splitFeature] != self._splitCondition

    class QuantNode(Node):
        """
        Subclass for Decision Tree Node Subclass; inherits attributes from Node class; used for quantitative data

        """

        def __init__(self, data, giniThreshold, indices, labels,
                     val_counts=None, prediction=None, gini=None
                    ):
            """
            Subclass Constructor for Node Subclass


            Parameters
            ~~~~~~~~~~~~~~~

            data [np.ndarray]: a reference pointer to a np.ndarray; training data
            giniThreshold [float]: The minimum improvement in Gini Impurity for a split to occur
            indices[list[int]]: a list of row indices that define the subset of _data and _labels for which the node is defined on
            labels [np.ndarray]: a refrence pointer to a np.ndarray; training labels
            val_counts[np.ndarray]: a np.ndarray created from using the np.unique(return_counts = True) method; Computed if not provided
            prediction[int/float/string]: The class prediction for the node; Computed if not provided
            gini [float]: the Gini Purity of the node; Computed if not provided


            Returns
            ~~~~~~~~~~~~~~~
            [QuantNode]: a Node Subclass object
        
            
            """
            # call super constructor
            # Time Complexity: 7l + 2i + 15
            # Space Complexity: 2l + i + 8
            super().__init__(data, giniThreshold, indices, labels,
                     val_counts, prediction, gini
                    )


        # function call
        # Time Complexity: 1
        # Space Complexity: 0
        def findBestSplit(self):
            """
            Subclass method that finds the best split on the node and returns information about it
            
            
            Parameters
            ~~~~~~~~~~~~~
            None
            
        
            Returns
            ~~~~~~~~~~~~~
            minGini [float]: The weighted Gini impurity of the child nodes
            bestSplit [float]: The optimal point to split on
            bestCol [int]: The column index that the split occurs on
            bestLeftIndices [list[int]]: a list of row indices that are defined in left child node
            bestRightIndices [list[int]]: a list of row indices that are defined in right child node
            bestLeftValCounts [np.ndarray]: np.ndarray created from using the np.unique(return_counts = True) method
            bestRightValCounts [np.ndarray]: np.ndarray created from using the np.unique(return_counts = True) method
            bestLeftGini [float]: the Gini Impurity of the left child
            bestRightGini [float]: the Gini Impurity of the left child


            """

            # initialize minimum gini
            # Time Complexity: 1
            # Space Complexity : 1
            minGini = np.inf

            # for column in the training data
            # Time Complexity: m (m updates to column)
            # Space Complexity: 1 (column)
            for column in range(self._data.shape[1]):

                # get the unique elements in workingData
                # Time Complexity: mi (loop thorugh WorkingData, m times)
                # Space Complexity: i
                workingData = self._data[self._indices, column]

                # get the unqiue values of workingData
                # Time Complexity : mi (iteration through data to get unqiue values, m times)
                # Space Complexity: i (i-1 midpoints, simplify to i)
                splitPoints = np.unique(workingData)


                # for each splitpoint
                # Time Complexity: mi (i-1 splitpoints, m times)
                # Space Complexity: 1 (midpoint)
                for point in splitPoints:

                    # Get left and right indices
                    # Time Complexity: 6im (2 boolena, 2 slices, 2 assignments, im times)
                    # Space Complexity: i (total indices of both add to i)
                    leftIndices = self._indices[workingData > point]
                    rightIndices = self._indices[workingData <= point]


                    # calculate leftValCounts and rightValCounts
                    # Time Complexity: 8im (2 total checks if value in unique list, 2 total counter updates, 4 slice, im times)
                    # Space Complexity : 4l (2l labels, 2l counts)
                    leftValCounts = np.unique(
                        self._labels[leftIndices],
                        return_counts=True
                    )

                    rightValCounts = np.unique(
                        self._labels[rightIndices],
                        return_counts=True
                    )

                    # calculaute left and right gini
                    # Time Complexity: im(2(5L +2) + 2) (2 computeGini, 2 assingment, im times)
                    # Space Complexity: 2
                    leftGini = self._computeGini(leftValCounts)
                    rightGini = self._computeGini(rightValCounts)
                    
                    
                    # calculaute child weights (ensures more equal splits)
                    # Time Complexity: 8im (4 len(), 2 division, 2 assignment, im times)
                    # Space Complexity: 2
                    leftWeight = len(leftIndices)/len(self._indices)
                    rightWeight = len(rightIndices)/len(self._indices)
                    
                    # calculaute mean Gini
                    # Time Complexity: 4im (1 sum, 2 multiplcation, 1 assignment, im times)
                    # Space Complexity: 1
                    newGini = (leftWeight*leftGini + rightWeight*rightGini)

                    # If newGini is better than the current gini
                    # Time Compelxity: im (im boolean)
                    # Space Complexity: 0
                    if newGini < minGini:
                    
                        # WORST CASE: im TIMES

                        # update maxGini, bestSplit, and best col
                        # Time Complexity: 3im
                        # Space Complexity: 2 (maxGini already initialized)
                        minGini = newGini
                        bestSplit = point
                        bestCol = column

                        # update left and right best indices
                        # Time Complexity: 2im
                        # Space Complexity: i (total i indices to partition)
                        bestLeftIndices = leftIndices
                        bestRightIndices = rightIndices

                        # update left and right best valcounts
                        # Time Complexity: 2im
                        # Space Complexity: 4l (2l labels, 2l counts)
                        bestLeftValCounts = leftValCounts
                        bestRightValCounts = rightValCounts

                        # update best left and right gini
                        # Time Complexity: 2im
                        # Space Complexity: 2
                        bestLeftGini = leftGini
                        bestRightGini = rightGini

            # return values
            # Time Complexity: 1
            # Space Complexity: 0
            return minGini, bestSplit, bestCol, bestLeftIndices, bestRightIndices, bestLeftValCounts, bestRightValCounts, bestLeftGini, bestRightGini

        # function call
        # Time Complexity: 1
        # Space Complexity: 0
        def _goLeft(self, data):
            """
            Subclass method that determines whether the test algorithm should traverse left

            Parameters
            ~~~~~~~~~~~~~~
            data [np.ndarray] : a 1xm test array
            
            
            Returns
            ~~~~~~~~~~~~~~
            [boolean]: Whether or not the test algorithm should traverse left
            
            
            """
            # return boolean
            # Time Complexity: 3 (1 slice, 1 boolean, 1 return)
            # Space Complexity: 0
            return data[self._splitFeature] > self._splitCondition

        
    def __init__(self, typeof, giniThreshold = None, maxDepth = None):
        """
        Constructor for Decision Tree Subclass


        Parameters
        ~~~~~~~~~~~~~~~
        _typeof [string]: the type of data to use for the decision tree. Must be 'Quant' or 'Qual'
        giniThreshold [float]: The minimum improvement in Gini Impurity for a split to occur
        maxDepth [int]: The maximum depth of the decision tree


        Returns
        ~~~~~~~~~~~~~~~
        [DecisionTreeClassifier]: a DecisionTreeClassifier object
        
        """
        
        # explicitly call superconstuctors
        ClassifierAlgorithm.__init__(self)
        Tree.__init__(self)
        
        # set type of tree (quant/qual)
        self._typeof = typeof
        
        # set hyperparameters
        if giniThreshold == None:
            self.giniThreshold = 0
        else:
            self.giniThreshold = giniThreshold
        
        self.maxDepth = maxDepth
    
    
    # fucntion call
    # time Complexity: 1
    # Space Complexity: 0
    def nodeWrapper(self, data, giniThreshold, indices, labels, 
                     val_counts = None, prediction = None, gini = None):
        """
        Wrapper method that returns a Node Sublcass based on _typeof
        
        
        Parameters
        ~~~~~~~~~~~~~~~
        data [np.ndarray]: a reference pointer to a np.ndarray; training data
        giniThreshold [float]: The minimum improvement in Gini Impurity for a split to occur
        indices[list[int]]: a list of row indices that define the subset of _data and _labels for which the node is defined on
        labels [np.ndarray]: a refrence pointer to a np.ndarray; training labels
        val_counts[np.ndarray]: a np.ndarray created from using the np.unique(return_counts = True) method; Computed if not provided
        prediction[int/float/string]: The class prediction for the node; Computed if not provided
        gini [float]: the Gini Purity of the node; Computed if not provided



        Returns
        ~~~~~~~~~~~~~~~
        [QuantNode/QualNode]: a Node Subclass object
        
        
        """
        
        # if type of is "Quant"
        # Time Complexity: 1
        # Space Complexity: 0
        if self._typeof == "Quant":

            # return node
            # Time Complexity: 7L + 2i + 15 + 1 (initalize, 1 return)
            # Space Complexity: 0
            return self.QuantNode(
                data,
                giniThreshold,
                indices,
                labels,
                val_counts,
                prediction,
                gini
            )
        
        # otherwise, if typeof is "Qual"
        elif self._typeof == "Qual":

            
            # return node
            # Time Complexity: 7L + 2i + 15 + 1 (initalize, 1 return)
            # Space Complexity: 0
            return self.QualNode(
                data,
                giniThreshold,
                indices,
                labels,
                val_counts,
                prediction,
                gini
            )
        
        
        # otherwise, throw exception
        else:
            raise TypeError("_typeof must be 'Quant' or 'Qual'; got " + str(self._typeof))
        
        
    # function call
    # Time Complexity: 1
    # Space Complexity: 0
    def train(self, data, labels, giniThreshold = None, maxDepth = None):
        """
        Train method that builds Decision Tree nodes
        
        Parameters
        ~~~~~~~~~~~~~~
        data [np.ndarray]: a nxm array used for training
        labels [np.ndarray]: a 1xn array representing the class labels of the training data
        giniThreshold [float]: The minimum improvement in Gini Impurity for a split to occur
        maxDepth [int]: The maximum depth of the decision tree
        
        Returns
        ~~~~~~~~~~~~~~
        [None]
        
        """
        
        # create label dictionary for probability mapping
        # Time complexity: 2L
        # Space Complexity: 2L
        self._labelDict = {lab: ind for ind, lab in enumerate(np.unique(labels))}
        
        
        # if giniThreshold not provided
        # Time Complexity: 1
        # Space Complexity: 0
        if giniThreshold == None:
            giniThreshold = self.giniThreshold
        
        
        # create root node
        # Time Complexity: 7L + 2i + 18 + 1 (nodeWrapper, 1 return)
        # Space Complexity: 2l + i + 8 (size of node)
        self.root = self.nodeWrapper(data, giniThreshold, np.arange(len(data)), labels)
        
        
        # if maxDepth is provided
        # Time Complexity: 1
        # Space Complexity: 0
        if maxDepth != None:
            self.maxDepth = maxDepth
        
        # Recursively build tree
        # WORST CASE: TREE has depth n
        # Time Complexity: n(45nm + 10lnm + 4n + 14L + m + 51)
        # Space Complexity: n(6n + 12l + 28)
        self._trainRecursive(self.root, 0)

        
    
    
    # function call
    # Time Complexity: 1
    # Space Compelxity: 0
    def _trainRecursive(self, node, depth):
        """
        Recursive train method that builds Decision Tree nodes
        
        Parameters
        ~~~~~~~~~~~~~~
        node [QuantNode/QualNode]: a Node subclass
        depth [int]: the current depth of the tree at current node
        
        Returns
        ~~~~~~~~~~~~~~
        [None]
        
        """
        
        # check if node is pure (exit condition)
        # Time Complexity: 3 (isPure)
        # Space Complexity: 0
        if node.isPure():
            # return
            # Time Complexity: 1
            # Space Complexity: 0
            return
        
        
        # if _maxDepth is provided
        # Time Complexity: 1
        # Space Complexity: 0
        if self.maxDepth != None:
            
            # If depth of tree is at the maximum allowed
            # Time Complexity: 1
            # Space Complexity: 0
            if depth >= self.maxDepth:
                
                # return
                # Time Complexity: 1
                # Space Complexity: 0
                return
        
        # Call findBestSplits
        # Time Complexity: 45im + 10lim + m + 3 (findBestSplits)
        # Space Complexity: 4i + 8L + 12 (findBestSplits)
        minGini, bestSplit, bestCol, bestLeftIndices, bestRightIndices, bestLeftValCounts, bestRightValCounts, bestLeftGini, bestRightGini = node.findBestSplit()
        
        
        
        # if gini split is significant
        # Time Complexity: 2
        # Space Complexity: 0 
        if (node._gini - minGini) > node._giniThreshold:  
            


            # create left child
            # Time Complexity: 7L + 2i + 18 + 1 (nodeWrapper, 1 return)
            # Space Complexity: 2l + i + 8 (size of node)
            node._leftChild = self.nodeWrapper(
                node._data,
                node._giniThreshold,
                bestLeftIndices,
                node._labels,
                bestLeftValCounts, 
                node._computePrediction(bestLeftValCounts),
                bestLeftGini
            )
            

            
            
            # create right child
            # Time Complexity: 7L + 2i + 18 + 1 (nodeWrapper, 1 return)
            # Space Complexity: 2l + i + 8 (size of node)
            node._rightChild = self.nodeWrapper(
                node._data,
                node._giniThreshold,
                bestRightIndices,
                node._labels,
                bestRightValCounts, 
                node._computePrediction(bestRightValCounts),
                bestRightGini
            )
            
            
            
            # set splitconidtion and feature
            # Time Complexity: 2
            # Space Complexity: 0 (already initialized)
            node._splitCondition = bestSplit
            node._splitFeature = bestCol
            
            
            # call train_recursive on left and right child
            # Time Complexity: 0 (defined already)
            self._trainRecursive(node._leftChild, depth+1)
            self._trainRecursive(node._rightChild, depth+1)
            
            
            
                
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    
    # Complexity Analysis
    
    # For reference:
        # n = number of rows
        # m = number of columns
        # l = number of unique labels
        # i = number of row indices defined on a node (each level of tree has indices summing to n)
    
    
    # for fixed m,l and variable n (number of rows in data)
        # Time Complexity : T(n) = 45mn^2 + 10lmn^2 + 4n^2 + 14ln + mn + 53n + 7l + 22

        # Let g(n) = n^2
        # let c = 11((4m + 10lm + 4) + 1)
        # Let n0 = 7l + 22
        # Then T(n) <= c*g(n) for all n >= n0 
        # Therefore T(n) has n^2 Big-Oh given that m,l is fixed

        # for fixed m,l, and variable n (number of rows in data)
        # Space Complexity : S(n) = 6n^2 + 12ln^2 + 29n + 2l + 8

        # Let g(n) = n^2
        # let c = 5(6 + 12l)
        # Let n0 = 2l + 8
        # Then S(n) <= c*g(n) for all n >= n0 
        # Therefore S(n) has n^2 Big - Oh given that m,l is fixed            
            
            
            
            
    # function call
    # Time Complexity: 1
    # Space Complexity: 0
    def test(self, data, probs = False):        
        """
        Recursive test method that predicts the class of test data
        
        Parameters
        ~~~~~~~~~~~~~~
        data [np.ndarray] : a jxm test array
        probs [boolean]; Whether the returned predictions should be a probability
        
        Returns
        ~~~~~~~~~~~~~~
        [np.ndarray]: an array of class predictions provided test data
        
        """
        
        # return list of predictions
        # Time Complexity: n(8n + 3) (_testRecursive, 1 slice, 1 varaible update, n times)
        # Space Complexity: n (list of length n)
        return np.array([self._testRecursive(self.root, data[i,:], probs) for i in range(len(data))])
            
            
    
    # function call
    # Time Complexity: 1
    # Space Complexity: 0
    def _testRecursive(self, node, data, probs):
        """
        Recursive test method that predicts the class of a test point
        
        Parameters
        ~~~~~~~~~~~~~~
        node [QuantNode/QualNode]: a Node subclass
        data [np.ndarray] : a 1xm test array
        probs [boolean]; Whether the returned predictions should be a probability
        
        Returns
        ~~~~~~~~~~~~~~
        node._prediction [int/string]: the class prediction of the test point
        
        """
        # if node doesn't have a split
        # Time Complexity: 4 (3 boolean, 1 return)
        # Space Complexity: 0
        if not node.hasChildren():
            
            # if probs are selected 
            # Time Complexity: 1
            # Space Complexity: 0
            if probs:
                
                # WONT BE EXECUTED FOR TIME COMPLEXITY ANALYSIS
                
                # get an empty class empirical distribtution
                probList = np.zeros(len(self._labelDict))
                
                #print(len(probList), probList)
                # get the sum of classes
                total = sum(node._val_counts[1])

                # for each class in valcounts
                for _,i in enumerate(node._val_counts[0]):
                      
                    #print(i, self._labelDict[i])
                    # add the probability of the class to the empirical distribtution
                    probList[self._labelDict[i]] = node._val_counts[1][_]/total
                    #print(len(probList), probList)
                    
                # return empirical distribution
                return probList
                
            else:
                # return predction
                # Time Complexity: 1
                # Space Complexity: 0
                return self._labelDict[node._prediction]
        
        # if path should traverse left
        # Time Complexity: 4
        # Space Complexity: 0        
        if node._goLeft(data):
            
            # recursive function call
            # Time Complexity: 0 (already calculated)
            # Space Complexity: 0
            return self._testRecursive(node._leftChild, data, probs)

        
        # otherwise, path should traverse right
        else:
            # recursive function call
            # Time Complexity: 0 (already calculated)
            # Space Complexity: 0
            return self._testRecursive(node._rightChild, data, probs)
        
        
        
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    
    # Complexity Analysis
    
    # For reference:
        # n = number of rows
    
    
    # for variable n (number of rows in data)
        # Time Complexity : T(n) = 8n^2 + 3n + 1

        # Let g(n) = n^2
        # let c = 12
        # Let n0 = 1
        # Then T(n) <= c*g(n) for all n >= n0 
        # Therefore T(n) has n^2 Big-Oh given that m,l is fixed

    # for variable n (number of rows in data)
        # Space Complexity : S(n) = n

        # Let g(n) = n
        # let c = 2
        # Let n0 = 1
        # Then S(n) <= c*g(n) for all n >= n0 
        # Therefore S(n) has linear Big - Oh given that m,l is fixed            
                  
        
        
    def _toString(self, node):
        """
        Recursive DFS method to create formatted string
        
        Parameters
        ~~~~~~~~~~~~~~
        node [QuantNode/QualNode]: a Node subclass
        
        
        Returns
        ~~~~~~~~~~~~~~
        [string]: a string representation of the Decision Tree
        
        """
        
        # if node is none, return empty string
        if node == None:
            return ''

        # call _toString on left and right child nodes
        lhs = self._toString(node._leftChild)
        rhs = self._toString(node._rightChild)
        
        
        # if node doesnt have a split, nodeVal is 'leaf'
        if node._splitCondition == None:
            nodeVal = 'leaf'
        
        # else if node is a float, nodeVal is '<####'
        elif type(node._splitCondition) == np.float64:
            nodeVal = '<' + '{:.4f}'.format(node._splitCondition)
        
        # otherwise, node is a string
        else:
            nodeVal = node._splitCondition
            
        # return the nodes splitting condition, if any, and the left and rhs strings, wrapped in brackets
        return '[' + nodeVal + ' ' + lhs + rhs + ']'
    


# In[2]:


class Experiment:
    """Class for evaluating the accuracy of models and alorithms from the ClassifierAlgorithm Subclasses



    Default Attributes
    ~~~~~~~~~~~~~~~
    _preds [None]: Holds the test prediction labels of the Classifier; When initialized in runCrossVal(), will hold a numpy.ndarray of shape (n,p) where n is the number of rows of data and p is the number of classifier algorithms passed to runCrossVal()


    """
    
    class HeapSort:
        """
        A Heap Sorting Algorithm Class, used to sort a 2 dimensional array by a selcted column index


        """
        def __init__(self, lst, on = 0):
            """
            Constructor for HeapSort class
            
            Paramters
            ~~~~~~~~~
            lst [np.ndarray]: A multidimensional array
            on [int]; The column index to sort on


            Returns
            ~~~~~~~
            [HeapSort]: a Heapsort Class Object 
            
            """

            # set the column to sort on
            self._on = on

            # heapify the list
            self.heapify(lst)



        def heapify(self, lst):
            """
            Transform lst to follow heap properties
            
            Paramters
            ~~~~~~~~~
            lst [np.ndarray]: A multidimensional array


            Returns
            ~~~~~~~
            None             
            
            
            """
            
            # only need to heap down on half the elements
            i = len(lst) // 2

            # deep copy of lst
            self.heapList = [ele for ele in lst]

            # set current size
            self.currentSize = len(lst)

            # iteratively heapdown
            while (i >= 0):
                self.heapDown(i)
                i -= 1


        def sort(self):
            """
            Iteratively unheap the Heap, returning a sorted descending list
            
            
            Paramters
            ~~~~~~~~~
            None


            Returns
            ~~~~~~~
            [np.ndarray]: The sorted array in descending order
            
            """

            # pop each value off heap, return sorted list
            return np.array([self._pop() for i in range(self.currentSize)])



        def heapDown(self, i):
            """
            Performs a heap down operation on a node
            
            Paramters
            ~~~~~~~~~
            i [int]: index of the heap


            Returns
            ~~~~~~~
            None           
            
            """
            
            # while a child of i exists
            while (2 * i + 1 ) < self.currentSize:

                # get the index of the minimum child
                mc = self.maxChild(i)

                # if parent is less than the child
                if self.heapList[i][self._on] < self.heapList[mc][self._on]:

                    # swap the parent and child
                    self.heapList[i], self.heapList[mc] = self.heapList[mc], self.heapList[i]

                # update i to the minimum child index
                i = mc

        def maxChild(self, i):
            """
            finds the index of the of the maximum child of node i
            
            Paramters
            ~~~~~~~~~
            i [int]: index of the heap


            Returns
            ~~~~~~~
            maxidx [int]: the index of the of the maximum child of node i         
            
            """

            # assume left child is min
            maxidx = 2 * i + 1
            maxchild = self.heapList[maxidx][self._on]

            # if right child exists
            if maxidx + 1 < self.currentSize:

                # if right child is greater than leftchild
                if self.heapList[maxidx + 1][self._on] > maxchild:

                    # min child index is right child
                    maxidx += 1

            # return index of minimum child
            return maxidx

        def _pop(self):
            """
            removes the top of the heap and returns it
            
            Paramters
            ~~~~~~~~~
            None


            Returns
            ~~~~~~~
            retval [np.ndarray]: the object popped from the top of the heap
              
            
            """

            # get the value that is going to be popped off
            retval = self.heapList[0]

            # replace the top of the heap with the bottom of the heap
            self.heapList[0] = self.heapList[self.currentSize - 1]

            # decrement current size
            self.currentSize -= 1

            # remove the last element in list (now duplicated to be in first index)
            self.heapList.pop() 

            # heap down on the first index
            self.heapDown(0)

            # return the popped value
            return retval
    
    # Default Attributes
    _preds = None
    _probs = None
    
    def __init__(self, dataset, cols, label, classifiers = []):
        """Instantiates the Experiment class;
        
        Paramters
        ~~~~~~~~~
        dataset [object]: An instantation of a Dataset Subclass
        cols [list[str],list[int],int]; The indices or column names of the dataset to train on 
        label [list[str],int]; The indice or column name of the label column
        classifiers [list[object]]: A list of instantiated objects that are ClassifierAlgorithm Subclasses 
        
        
        Returns
        ~~~~~~~
        Experiment [object] 
        
        """
        
        # dictionary of compatible classifier-dataset methods
        compatible_methods = {
            "<class 'DataToolbox.simpleKNNClassifier'>" : [
                "<class 'DataToolbox.QuantDataSet'>"
            ],
            "<class 'DataToolbox.DecisionTreeClassifier'>" : [
                "<class 'DataToolbox.QuantDataSet'>",
                "<class 'DataToolbox.QualDataSet'>"
            ],
            "<class 'DataToolbox.kdTreeKNNClassifier'>" : [
                "<class 'DataToolbox.QuantDataSet'>"
            ]
        }

        for classifier in classifiers:
            if str(type(dataset)) not in compatible_methods[str(type(classifier))]:
                raise TypeError(str(type(classifier)) + 'not compatible with ' + str(type(dataset)))
        
        # get the indices of the data and labels 
        data_ind = dataset._getColIdx(include = cols)
        label_ind = dataset._getColIdx(include = label)
        
        
        # assign attributes
        self._data = dataset._data[:,data_ind]
        self._labels = dataset._data[:,label_ind].flatten()
        self._classifiers = classifiers
        
        
        # get the number of unique labels in the data for use in ConfusionMatrix()
        self._uniquelabels = len(set(self._labels))
        
        # dictionary mapping labels to an index
        self._labelDict = {lab: ind for ind, lab in enumerate(np.unique(self._labels))}
        self._labelDictBackwards = {ind: lab for ind, lab in enumerate(np.unique(self._labels))}
        
        # get number of rows in the data
        self._nrow = self._data.shape[0]

        
    
    def runCrossVal(self, kFolds = 5, probs = False):
        """Member Function; performs cross validation on an a selection of ClassifierAlgorithm subclass objects
        
        
        Paramters
        ~~~~~~~~~
        kFolds [int]; The number of partitions to use for cross validation; Default: 5
        probs [boolean]; Whether the returned predictions should be a probability
        
        Returns
        ~~~~~~~
        None
        
        """
        # kFolds is not a postive integer, raise an error
        if type(kFolds) != int or kFolds < 1:
            raise ValueError('kFolds must be a positive integer; got ' + str(kFolds))
        
        
        
        # get the number of rows in the data
        #nrow = self._data.shape[0]
        
        if probs:

            self._probs = np.zeros(
                (self._nrow, self._uniquelabels*len(self._classifiers))
            )
        else:
            # initialize empty array of shape (nrow, len(classifiers))
            self._preds = np.zeros(
                (self._nrow, len(self._classifiers))
            )
        
        # get the number of rows that will be in a single partition
        partition = (self._nrow//kFolds) + 1
        
        # create array of indices ranging from 0 to nrow
        shuff = [i for i in range(self._nrow)]
        
        # randomize the array
        np.random.shuffle(shuff)
        
        # for each fold
        for fold in range(kFolds):
            
            print("Computing Fold: ", fold+1)
            
            # extract train data by subsetting data not in the defined fold
            traindata = np.ndarray.copy(
                np.vstack(
                    [
                        self._data[shuff[:fold*partition], :],
                        self._data[shuff[(fold + 1)*partition:], :],

                    ]
                )
            )
            
            # extract train labels by subsetting labels not in the defined fold
            trainlabels = np.ndarray.copy(
                    np.concatenate(
                    [
                        self._labels[shuff[:fold*partition]],
                        self._labels[shuff[(fold + 1)*partition:]],

                    ],
                    axis = 0
                )
            )
            
            # extract test data by subsetting data in the defined fold
            testdata = self._data[shuff[fold*partition: (fold+1)*partition],:]
            

            
            # for each classifier
            for _,classifier in enumerate(self._classifiers):



                # train the data on the classifier
                classifier.train(traindata, trainlabels)
                
                # if return probabilities
                if probs:

                    # test the test data on the trained classifier, get the label predictions
                    probDist = classifier.test(testdata, probs = True)

                    # place the predicted probabilties in the correct indices of the probs matrix    
                    self._probs[:, _*self._uniquelabels : (_+1)*self._uniquelabels: ][shuff[fold*partition: (fold+1)*partition]] = probDist


                else:
                    # test the test data on the trained classifier, get the label predictions
                    preds = classifier.test(testdata)


                    # place the predicted labels in the correct indices of the preds matrix
                    self._preds[shuff[fold*partition: (fold+1)*partition], _ ] = preds
        
        
        pass
    
    
    

    
    def score(self):
        """Member Function; calculates accuracy of ClassifierAlgorithm subclasses on the data; prints a table

        Paramters
        ~~~~~~~~~
        None
        
        
        Returns
        ~~~~~~~
        None
        
        """
        # for reference:
            # self._preds has shape (n,k)
            # k represents the the number of classifiers
        
        
        # if preds is not instantiated, raise an error
        # Time Complexity : 2 (1 get__attr__ method, 1 boolean comparison)
        # Space Complexity: 0 
        if not isinstance(self._preds, np.ndarray):
             raise NotImplementedError(
                'Cross Validation has not been executed; execute runCrossVal() method first'
            )
                
        # get the Class name of each classifier in self._classifiers
        # Time Complexity : 5k (1 get__attr__ method, 1 cast as string, 1 split, 2 slices, k times)
        # Space Complexity: k (list of length k created)
        classifiernames = [str(type(classifier)).split('.')[-1][:-2] for classifier in self._classifiers]
        
        # initialize empty list of accuracies
        # Time Complexity : 1 (1 instantiation)
        # Space Complexity: 0 (empty array)        
        accuracies = []
        
        
        # for each classifier
        # Time Complexity : 2k (i updated k times, k boolean comparisons)
        # Space Complexity: 1 (hold i in memory)        
        for i in range(len(self._classifiers)):
            
            # append the accuriacies of each classifier
            # Time Complexity : k(4n+1) (2n indexes n  boolean comparisons, sum of n values, one division, k times)
            # Space Complexity: k (append k values to list)    
            accuracies.append(
                sum(
                    [
                        self._preds[j,i] == self._labelDict[self._labels[j]] for j in range(self._nrow)
                    ]
                )/self._nrow
            )
        
        # print table header and divider
        # Time Complexity : 2
        # Space Complexity: 0
        print ('{:<35} {:<10}'.format('Classifier Name','CV Accuracy'))
        print('~'*50)
        
        # for each classifier
        # Time Complexity : k
        # Space Complexity: 0 (i already allocated)        
        for i in range(len(classifiernames)):
            # print row of table
            # Time Complexity : k
            # Space Complexity: 0  
            print ('{:<35} {:<10}'.format(classifiernames[i], accuracies[i]))
        
        pass 
    
        ##############################
        # Complexity Analysis
        #############################
        
        
        # for reference:
            # self._preds has shape (n,k)
            # k represents the the number of classifiers
        
  
        # for fixed k and variable n (number of rows in data)
        # Time Complexity : T(n) = 4kn + 10k + 4

        # Let g(n) = n
        # let c = 4k +1
        # Let n0 = 10k + 4
        # Then T(n) <= c*g(n) for all n >= n0 
        # Therefore T(n) has linear Big-Oh given that k is fixed

        # for fixed k, and variable n (number of rows in data)
        # Space Complexity : S(n) = 2k + 1

        # Let g(n) = 1
        # let c = 2k +1
        # Let n0 = 1
        # Then S(n) <= c*g(n) for all n >= n0 
        # Therefore S(n) has constant Big - Oh given that k is fixed
    
    def confusionMatrix(self):
        """Private Member Function; calculates a confusion matrix for ClassifierAlgorithm subclasses and plots them
 
        Paramters
        ~~~~~~~~~
        None
        
        
        Returns
        ~~~~~~~
        None
        
        """
        
        # for reference:
            # self._preds has shape (n,k)
            # k represents the the number of classifiers
            # l represents the number of unique labels
        
        
        # if preds is not instantiated, raise an error
        # Time Complexity : 2 (1 get__attr__ method, 1 boolean comparison)
        # Space Complexity: 0         
        if not isinstance(self._preds, np.ndarray):
            raise NotImplementedError(
                'Cross Validation has not been executed; execute runCrossVal() method first'
            )
        
        # get list of true labels for the data
        # Time Complexity: 2n (n variable castings, n iterations)
        # Space Complexity : n (n labels)
        labs = [self._labelDict[i] for i in self._labels]
        
        # for each classifier
        # Time Complexity: 3k (2k assingments of _ and classifier, k boolean comparisons)
        # Space Complexity: 2 (_ and classifier)
        for _,classifier in enumerate(self._classifiers):
            
            # get list of predicted labels for the classifier
            # Time Complexity: 2nk (n variable castings, n iterations, k times)
            # Space Complexity : n (n labels, reallocated k times)
            preds = [int(i) for i in self._preds[:,_]]
            
            # lxl matrix of zeros
            # Time Complexity: kl^2
            # Space Complexity: L^2
            heatmap = np.zeros((self._uniquelabels, self._uniquelabels))
            
            # for each row in data
            # Time Complexity: 2kn (n assingments of i, n boolean comparisons)
            # Space Complexity: 1 (update i)            
            for i in range(self._nrow):
                
                # add 1 to the correpsonding indice of matrix
                # Time Complexity: 4kn (3kn slices, 4kn summations)
                # Space Complexity: 0 (heatmap already allocated) 
                heatmap[labs[i],preds[i]] += 1
            
            
            ###########################################
            #
            # Remaining section will not be computed for Complexity 
            # Analysis, heatmap for each classifier has been generated
            #
            ###########################################
            
            # set up subplots
            fig, ax = plt.subplots(figsize = (15,10))
            
            # plot heatmap with value counts annotated
            sns.heatmap(
                heatmap,
                annot = True,
                fmt = '.0f',
                xticklabels = [self._labelDictBackwards[i] for i in range(self._uniquelabels)],
                yticklabels = [self._labelDictBackwards[i] for i in range(self._uniquelabels)],
                ax = ax
            )
            # set x and y labels
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            
            # set figure title with Classifier name
            fig.suptitle(str(type(classifier)).split('.')[-1][:-2])
            
            # show plot
            plt.show()
    
        ##############################
        # Complexity Analysis
        #############################
        
        
        # for reference:
            # self._preds has shape (n,k)
            # k represents the the number of classifiers
            # l represents the number of unique labels
        
  
        # for fixed k.l and variable n (number of rows in data)
        # Time Complexity : T(n) = 8kn + 2n + k(l)^2 + 3k + 2

        # Let g(n) = n
        # let c = 8k +3
        # Let n0 = kL^2 + 3k + 2
        # Then T(n) <= c*g(n) for all n >= n0 
        # Therefore T(n) has linear Big-Oh given that k,l is fixed

        
        # for fixed k,l and variable n (number of rows in data)
        # Space Complexity : S(n) = 2n + l^2 + 3

        # Let g(n) = 1
        # let c = 3
        # Let n0 = l^2 + 3
        # Then S(n) <= c*g(n) for all n >= n0 
        # Therefore S(n) has linear Big - Oh given that k,l is fixed
        
        
    # function call
    # Time Complexity: 1
    # Space Complexity: 0
    def ROC(self):
        """
        Computes and plots the ROC curve for 2 class and mulitclass problems for each classifier
        
        Paramters
        ~~~~~~~~~
        None
        
        
        Returns
        ~~~~~~~
        None        
        
        
        """
        
        # if probs is not instantiated, raise an error
        # Time Complexity : 2 (1 get__attr__ method, 1 boolean comparison)
        # Space Complexity: 0         
        if not isinstance(self._probs, np.ndarray):
            raise NotImplementedError(
                'Cross Validation has not been executed; execute runCrossVal(probs = True) method first'
            )
        
        # set up subplots
        # FOR VISUALIZATION
        # NOT COMPUTED FOR COMPLEXITY ANALYSIS
        fig, ax = plt.subplots(figsize = (15,10))
        
        # tab10 colormap
        # FOR VISUALIZATION
        # NOT COMPUTED FOR COMPLEXITY ANALYSIS
        cmap = [
            '#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#d62728',
            '#9467bd',
            '#8c564b',
            '#e377c2',
            '#7f7f7f',
            '#bcbd22',
            '#17becf'
        ]
        
        # plot line
        # FOR VISUALIZATION
        # NOT COMPUTED FOR COMPLEXITY ANALYSIS        
        sns.lineplot(
            x = [0,1],
            y = [0,1],
            ax = ax,
            color = '#000000'
        )
        
        
        # concatenate the labels with the prediction probabilities
        # Time Complexity: 2n (iterate through flattened list, dictionaty lookup)
        # Space Complexity: nl + n (rows times number of classes + label array)
        ROCdata = np.hstack(
            [
                self._probs,
                np.array([self._labelDict[i] for i in self._labels.flatten()]).reshape(-1,1)
            ]
        )
        
        
        # if its a two class problem, set loop range
        # Time Complexity: 1
        # Space Complexity: 0
        if self._uniquelabels == 2:
            # set looprange
            # Time Complexity: 1
            # Space Complexity: 0
            loopRange = 1
        # otherwise, multiclass problem
        else:
            # set looprange
            # Time Complexity: 1
            # Space Complexity: 0
            loopRange = self._uniquelabels
        
        
        # for each classifer
        # Time Complexity: 2c (c updates to _, c updates to classifier)
        # Space Complexity: 2 (2 values)
        for _,classifier in enumerate(self._classifiers):
            
            
            # for each loop in looprange:
            # Time Complexity: lc (l iterations, c times)
            # Space Complexity: 1 (loop)
            for loop in range(loopRange):
                
                # Sort ROCdata using a heapsort by current indice
                # Time Complexity: lc(n log n) (heapsort, lc times)
                # Space Complexity: 0
                ROCdata = self.HeapSort(ROCdata, on = (_ * self._uniquelabels) + loop).sort()
                
                # get total number of postive and negative cases for class 'loop'
                # Time Compelxity: lc(2n + 4) (n boolean, n sums, 2 assingments, 1 subtraction, 1 len, lc times)
                # Space Complexity: 2
                P = sum(ROCdata[:,-1] == loop)
                N = len(ROCdata) - P
                
                
                # initialize FP and TP
                # Time Compelxity: 2lc
                # Space Complexity: 2                       
                FP = TP = 0
                
                # initialize TPR and FPR
                # Time Compelxity: 2lc
                # Space Complexity: 2
                TPR = [] 
                FPR = []

                # get total number of postive and negative cases for class 'loop'
                # Time Compelxity: lc
                # Space Complexity: 1
                prev =  -np.inf
                
                # for each row in ROCdata
                # Time Complexity: lcn (n updates, lc times)
                # Space Complexity: 1
                for i in range(len(ROCdata)):
            
                    # if probability is not the same as the previous
                    # Time Complexity: 4lcn (1 multiplication, 1 sum, 1 slice, 1 boolean, lcn times)
                    # Space Complexity: 0
                    if ROCdata[i,(_ * self._uniquelabels) + loop] != prev:

                        # append TP rate and FP rate to respctive lists
                        # Time Complexity: 4lcn (2 appends, 2 divsions, lcn times)
                        # Space Complexity: 2n
                        TPR.append(TP/P)
                        FPR.append(FP/N)

                        # update previoes probability
                        # Time Complexity: 4lcn (1 multiplication, 1 sum, 1 slice, 1 assignment, lcn times)
                        # Space Complexity: 0 
                        prev = ROCdata[i,(_ * self._uniquelabels) + loop]

                        
                    # increment TP or FP based on if class label matches loop 
                    # Time Complexity: 3lcn (1 slice, 1 boolean, 1 update)
                    # Space Complexity: 0
                    if ROCdata[i,-1] == loop:
                        TP += 1
                    else:
                        FP += 1
                
                # if probability is not the same as the previous
                # Time Complexity: 4 (2 division, 2 append)
                # Space Complexity: 2
                TPR.append(TP/P)
                FPR.append(FP/N)
                
                
                # plot line
                # FOR VISUALIZATION
                # NOT COMPUTED FOR COMPLEXITY ANALYSIS 
                sns.lineplot(
                    x = FPR,
                    y = TPR,
                    ax = ax,
                    palette = (_ * self._uniquelabels) + loop,
                    label = str(type(classifier)).split('.')[-1][:-2] + ' : ' + str(self._labelDictBackwards[loop]) 
                )
        
        
        # set plot attributes
        # FOR VISUALIZATION
        # NOT COMPUTED FOR COMPLEXITY ANALYSIS 
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_ylabel("True Positive Rate", fontsize = 14)
        ax.set_xlabel("False Positive Rate", fontsize = 14)
        fig.suptitle("ROC Scores", fontsize = 16)
        plt.legend(loc='lower right')
        plt.show()
        
        
        ##############################
        # Complexity Analysis
        #############################
        
        
        # for reference:
            # n represents the number of predcitions
            # c represents the number of classifiers
            # l represents the number of unique labels
        
  
        # for fixed l,c and variable n (number of rows in data)
        # Time Complexity : T(n) = lc(nlogn) + 14lcn + 2n + 10lc + 2c + 9

        # Let g(n) = nlogn
        # let c = 15lc + 3
        # Let n0 = 10lc + 2c +9
        # Then T(n) <= c*g(n) for all n >= n0 
        # Therefore T(n) has nlogn Big-Oh given that k,l is fixed

        
        # for fixed k,l and variable n (number of rows in data)
        # Space Complexity : S(n) = (l+3)n + 12

        # Let g(n) = n
        # let c = l + 4
        # Let n0 = 12
        # Then S(n) <= c*g(n) for all n >= n0 
        # Therefore S(n) has linear Big - Oh given that k,l is fixed        


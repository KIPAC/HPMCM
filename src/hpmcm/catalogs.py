
import pandas


class BaseDataFrameReducer:

    def __init__(self, **kwargs):
        self.config = kwargs.copy()


    def __call__(self, inputData: pandas.DataFrame) -> pandas.DataFrame:
        raise NotImplementedError()



class MetaDetectDataFrameReducers(BaseDataFrameReducer):

    def __init__(self, **kwargs):
        BaseDataFrameReducer.__init__(self, **kwargs)
        self.config.setdefault('catType', 'wmom')
        self.config.setdefault('snrCut', 1.0)
    
    def __call__(self, inputData: pandas.DataFrame) -> pandas.DataFrame:

        idx_x = 20*inputData['patch_x'].values + inputData['cell_x'].values
        idx_y = 20*inputData['patch_y'].values + inputData['cell_y'].values
        cent_x = 150*idx_x -75
        cent_y = 150*idx_y -75

        local_x = inputData['col'] - cent_x
        local_y = inputData['row'] - cent_y

        inputData["local_x"] = local_x
        inputData["local_y"] = local_y

        if self.config['catType'] == "wmom":
            inputData["SNR"] = inputData["wmom_band_flux_r"] / inputData["wmom_band_flux_err_r"]
        elif self.config['catType'] == "gauss":
            inputData["SNR"] = 1.0 / inputData["gauss_mag_r_err"]
        
        inputDataCleaned = inputData[(inputData.SNR > self.config['snrCut'])]        
        outputData = inputDataCleaned.copy(deep=True)
        
        return outputData




    


    
        



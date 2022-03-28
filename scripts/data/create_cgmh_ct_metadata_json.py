import pydicom, os
from tqdm import tqdm
import json
import pandas as pd
from ast import literal_eval
from datetime import datetime

DIR = "/home/images/CGMH_LDCT"

dicoms = []
for root, _, files in os.walk(DIR):
    dicoms.extend([ os.path.join(root, f) for f in files ])  #extend( [os.path.join(root, f) for f in files] )

metadata = pd.read_csv('/home/images/MIT_LDCT.csv')
#Pseudo ID,Dicom count,Date (Ca Dx or Last FU),"Ca (Lung, other, none)"

dataset = []
pidindex = {}
eidindex = {}
sidindex = {}

for path in tqdm(dicoms, total=len(dicoms)):
    try:
        dcm = pydicom.dcmread(path, stop_before_pixels = True)
        studydate = dcm.StudyDate
        year = studydate[:4]
        manufacturer = dcm.Manufacturer
        pid = dcm.PatientID
        examid = dcm.StudyInstanceUID
        seriesid = dcm.SeriesInstanceUID
        position = float(dcm.ImagePositionPatient[-1])
        imagetype=dcm.ImageType
        #position = float(literal_eval(dcm.ImagePositionPatient)[-1])

        metarow = metadata[metadata['Pseudo ID'] == pid].to_dict('report')
        assert len(metarow) == 1
        metarow = metarow[0]
        cancer = metarow['Ca (Lung, other, none)'].lower()
        censor_date = str(metarow['Date (Ca Dx or Last FU)'])

        cumulative_eid = '{}_{}'.format(pid, examid)
        cumulative_sid = '{}_{}'.format(cumulative_eid, seriesid)

        sdict = {
                'seriesid': seriesid,
                'ImageType': imagetype,
                'slice_thickness': float(dcm.SliceThickness),
                'pixel_spacing': [float(d) for d in  dcm.PixelSpacing],
                'slice_position': [position],
                'paths': [path]
                }

        edict = {
                'examid': examid,
                'manufacturer': manufacturer,
                'year': year,
                'study_date': studydate,
                'series': [sdict],
                'cancer': cancer,
                'days_to_event': (datetime.strptime(censor_date, '%Y%m%d') - datetime.strptime(dcm.StudyDate,'%Y%m%d')).days
                }

        pdict = { 
                'pid': pid,
                'exams': [edict],
                'split': 'test',
                'cancer': cancer,
                'censor_date': censor_date
                }

        if pid in pidindex:
            if cumulative_eid in eidindex:
                if cumulative_sid in sidindex:
                    dataset[ pidindex[pid] ]['exams'][ eidindex[cumulative_eid] ]['series'][sidindex[cumulative_sid]]['slice_position'].append(position)
                    dataset[ pidindex[pid] ]['exams'][ eidindex[cumulative_eid] ]['series'][sidindex[cumulative_sid]]['paths'].append(path)
                else: # new series
                    sidindex[cumulative_sid] = len( dataset[ pidindex[pid] ]['exams'][ eidindex[cumulative_eid] ]['series'] )
                    dataset[ pidindex[pid] ]['exams'][ eidindex[cumulative_eid] ]['series'].append(sdict)
            else: # new exam
                eidindex[cumulative_eid] = len( dataset[ pidindex[pid] ]['exams'] )
                dataset[ pidindex[pid] ]['exams'].append(edict)

        else:
            pidindex[pid] = len(dataset)
            eidindex[cumulative_eid] = 0
            sidindex[cumulative_sid] = 0

            dataset.append(pdict)

    except:
        pass

print()
json.dump(dataset, open('ldct_dataset.json', 'w'))

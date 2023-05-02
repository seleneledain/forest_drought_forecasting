import os, json, logging, sys, traceback, asyncio, aiohttp, warnings
from typing import Dict, Tuple, List, Union, Any
from abc import ABC, ABCMeta, abstractmethod
from urllib.parse import urljoin
from pathlib import Path
from time import time
from collections import namedtuple

import ibm_boto3
from ibm_botocore.client import Config as IBMConfig
from ibm_botocore.client import ClientError as IBMClientError

logger = logging.getLogger(__name__)

################################################################
# Global variables
################################################################

MIN_STATUS_INTERVAL = 10
MAX_WORKERS = 8


################################################################
# Functions wrapping PAIRS upload API
################################################################=

async def startPairsUpload(session: aiohttp.ClientSession, uploadInfo: Dict, pairsHost: str = 'https://pairs.res.ibm.com') -> Union[str, None]:
    '''
    Tracks the status of the upload with upload ID uploadID every statusInterval seconds.
    :param session:        aiohttp.ClientSession object. This will be constructed with
                           your PAIRS credentials.
    :param uploadID:       ID of a previously initialized upload as returned by the
                           PAIRS API.
    :param statusInterval: Minimum wait time between upload status queries.
    :param pairsHost:      Scheme and URL of IBM PAIRS installation. E.g.
                           https://pairs.res.ibm.com.
    
    :return:               Tuple uploadSucceeded, uploadStatus. The former is a boolean
                           variable indicating overall success, the latter the response
                           given by the upload API.
    '''
    url = urljoin(pairsHost, '/v2/uploader/upload')
    logger.debug('Submitting upload request to PAIRS.')
    async with session.post(url, json=uploadInfo) as resp:
        if resp.status == 201:
            logger.debug('Upload request submitted to PAIRS. Status 201.')
            respJson = await resp.json()
            logger.debug('Upload request submitted. JSON response decoded.')
            uploadID = respJson.get('id')
            logger.debug('Registered PAIRS upload {}.'.format(uploadID))
            return uploadID
        elif resp.status == 400:
            # Incorrect upload definition.
            logger.error('Unable to initialize upload - incorrect upload definition.')
            return None
        elif resp.status == 401:
            # Missing authentication.
            logger.error('Unable to initialize upload - no authentication provided.')
            return None
        elif resp.status == 403:
            # Not authorized.
            logger.error('Unable to initialize upload - not authorized.')
            return None
        else:
            logger.error('Unable to initialize upload - HTTP response code {}.'.format(resp.status))
            return None
        
async def trackPairsUpload(session: aiohttp.ClientSession, uploadID: str, statusInterval: int=60, pairsHost: str = 'https://pairs.res.ibm.com') -> Tuple[bool, Dict]:
    '''
    Tracks the status of the upload with upload ID uploadID every statusInterval seconds.
    :param session:        aiohttp.ClientSession object. This will be constructed with
                           your PAIRS credentials.
    :param uploadID:       ID of a previously initialized upload as returned by the
                           PAIRS API.
    :param statusInterval: Minimum wait time between upload status queries.
    :param pairsHost:      Scheme and URL of IBM PAIRS installation. E.g.
                           https://pairs.res.ibm.com.
    
    :return:               Tuple uploadSucceeded, uploadStatus. The former is a boolean
                           variable indicating overall success, the latter the response
                           given by the upload API.
    '''
    if statusInterval < MIN_STATUS_INTERVAL:
        raise ValueError('\'statusInterval\' has to be at least {} seconds.'.format(MIN_STATUS_INTERVAL))
    url = urljoin(pairsHost, '/v2/uploader/upload/{uID}'.format(uID=uploadID))
    while True:
        logger.debug('Checking PAIRS upload status {}.'.format(uploadID))
        async with session.get(url) as resp:
            if resp.status == 200:
                uploadStatus = await resp.json()
                if uploadStatus['status'] == 'SUCCEEDED':
                    logger.debug('PAIRS upload {} succeeded.'.format(uploadID))
                    return True, uploadStatus
                elif uploadStatus['status'] == 'FAILED':
                    logger.debug('PAIRS upload {} failed.'.format(uploadID))
                    return False, uploadStatus
                else:
                    for singleUploadStatus in uploadStatus['summary']:
                        if singleUploadStatus['status'] < 0:
                            logger.debug('PAIRS upload {} failed.'.format(uploadID))
                            return False, uploadStatus
            elif resp.status == 400:
                # Cannot identify upload with tracking ID
                logger.error('Unable to track upload - invalid tracking ID.')
                pass
            elif resp.status == 401:
                # No basic authentication provided
                logger.error('Unable to track upload - no authentication provided.')
                pass
            else:
                logger.error('Unable to track upload - HTTP response code {}.'.format(resp.status))
                pass
        await asyncio.sleep(statusInterval)
        
async def runPairsUpload(session: aiohttp.ClientSession, uploadInfo: Dict, statusInterval: int=60, timeout: int=2000, pairsHost: str = 'https://pairs.res.ibm.com') -> Tuple[bool, Dict]:
    '''
    Uploads a single data file from Cloud Object Store (COS) to IBM PAIRS.
    :param session:        aiohttp.ClientSession object. This will be constructed with
                           your PAIRS credentials.
    :param uploadInfo:     JSON (or rather python dictionary) defining the upload. The
                           format of this JSON is defined by the PAIRS upload API.
    :param statusInterval: Minimum wait time between upload status queries.
    :param timeout:        Timeout for an upload to be considered as failed.
    :param pairsHost:      Scheme and URL of IBM PAIRS installation. E.g.
                           https://pairs.res.ibm.com.
    
    :return:               Tuple uploadSucceeded, uploadStatus. The former is a boolean
                           variable indicating overall success, the latter the response
                           given by the upload API.
    '''
    if statusInterval < MIN_STATUS_INTERVAL:
        raise ValueError('\'statusInterval\' has to be at least {} seconds.'.format(MIN_STATUS_INTERVAL))
    if timeout <= 0:
        raise ValueError('\'timeout\' has to be a positive integer.')
    logger.debug('Starting PAIRS upload.')
    uploadStartTime = time()
    uploadID = await startPairsUpload(session, uploadInfo, pairsHost)
    if uploadID:
        uploadSucceeded, uploadStatus = await asyncio.wait_for(trackPairsUpload(session, uploadID, statusInterval, pairsHost), timeout=timeout)
        logger.debug('Completed PAIRS upload in {} sec.'.format(round(time() - uploadStartTime)))
    else:
        uploadSucceeded, uploadStatus = False, dict()
    return uploadSucceeded, uploadStatus


################################################################
# Objects wrapping Cloud Object Storage
################################################################

CosObject = namedtuple('CosObject', ['key', 'size'])

class CosBucket(metaclass=ABCMeta):
    '''
    Cloud Object Storage bucket used to facilitate uploads to IBM PAIRS Geoscope.
    The bucket serves as an intermediate staging area when uploading
    files from a local source. Alterntively, data already present in the bucket
    can be ingested into PAIRS directly.
    '''
    
    @abstractmethod
    def upload(self, fileName: Union[Path, str], objectName: str):
        '''
        Uploads data file `fileName` and meta file `fileName.meta.json` to
        Cloud Object Storage (COS). Will not upload if the meta file does
        not exist. The data file will be pushed to `objectName`, the meta
        data file to `objectName.meta.json`. The default value for `objectName`
        is `fileName.name`.
        '''
        pass
    
    @abstractmethod
    def cosObjects(self) -> Dict[str, CosObject]:
        '''
        Returns the internal dictionary of CosObjects that would be uploaded
        if using PAIRSUpload. Note that this is not the complete lists of
        objects in the bucket, but usually a subset. The dictionary is indexed
        by the object's key.
        '''
        pass
    
    @abstractmethod
    def getMetaData(self, objectName: str) -> Dict:
        '''
        Returns the PAIRS upload meta data JSON for the object `objectName`.
        '''
        pass
    
    @abstractmethod
    def generatePresignedURL(self, objectName: str, expirationTime: int) -> str:
        '''
        Returns a presigned URL for the COS object `objectName`. The PAIRS
        uploader can then use this URL to upload to IBM PAIRS Geoscope.
        '''
        pass
    
    @abstractmethod
    def delete(self, objectName: str):
        '''
        Delete object `objectName` as well as the associated meta file from
        COS. Deletion will only happen if the object is part of `cosObjects`.
        Moreover, after deletion it will be removed from `cosObjects`.
        '''
        pass

class IBMCosBucket(CosBucket):
    '''
    IBM Cloud Object Store (COS) bucket used to upload to IBM PAIRS Geoscope.
    The bucket can serve as a staging area for uploads form a local source.
    Alternatively is is also possible to ingest data already located in the
    bucket. In both cases, an instance of this class should be used to intialize
    a PAIRSUpload object.
    
    The various keys needed to construct this can be obtained by logging into
    your IBM Cloud account at https://cloud.ibm.com.
    
    See https://www.ibm.com/cloud/object-storage for details on IBM COS.
    
    Usage: In most cases it is sufficient to construct an instance and hand
    this over to PAIRSUpload. See the documentation of various class methods
    for details on advanced usage.
    '''
    
    def __init__(self, apikey: str, resourceInstanceID: str, authEndpoint: str, endpoint: str, bucketName: str, accessKeyID: str=None, secretAccessKey: str=None):
        self.apikey = apikey
        self.resourceInstanceID = resourceInstanceID
        self.authEndpoint = authEndpoint
        self.endpoint = endpoint
        self.bucketName = bucketName
        
        self.cosResource = IBMCosBucket.createIBMCosResource(apikey, resourceInstanceID, authEndpoint, endpoint)
        if accessKeyID and secretAccessKey:
            self.cosHMACClient = IBMCosBucket.getHMACClient(accessKeyID, secretAccessKey, resourceInstanceID, authEndpoint, endpoint)
        else:
            logger.warning('Unable to create service client without \'accessKeyID\' and \'secretAccessKey\'.')
            self.cosHMACClient = None
                
        self.cosBucket = self.cosResource.Bucket(bucketName)
        
        self._cosObjects = dict()
    
    def cosObjects(self) -> Dict[str, CosObject]:
        '''
        Returns a dictionary containing all objects that will be ingested into
        IBM PAIRS if PAIRSUpload.run() is triggered.
        '''
        return self._cosObjects
    
    def updateCosObjects(self, **kwargs):
        '''
        Selects objects already present in the bucket. These objects
        can later be ingested into IBM PAIRS. Call cosObjects() to obtain the
        set of objects currently staged for ingestion.
        
        All keyword arguments **kwargs are used to filter the objects
        in the bucket. Currently the IBM COS API allows the following:
        
        :param Delimeter:
        :type Delimeter:          str
        :param Encoding:
        :type Encoding:           str
        :param Marker:
        :type Marker:             str 
        :param MaxKeys:           Maximum number of objects selected. Note that
                                  the objects selected are further filtered. Only
                                  those for which a .meta.json file has been
                                  selected as well will be included in a
                                  subsequent upload. Thus, selecting a small value
                                  for MaxKeys might lead to a much smaller number
                                  of objects ingested into IBM PAIRS.
        :type MaxKeys:            int
        :param MirrorDestination: 
        :type MirrorDestination:  str
        :param Prefix:            Only objects with this prefix will be used.
        :type Prefix:             str
        :param RequestPayer:
        :type RequestPayer:
        '''
        allCosObjects = [o for o in self.cosBucket.objects.filter(**kwargs)]
        metaObjectKeys = [o.key for o in allCosObjects if o.key.endswith('.meta.json')]
        dataObjectKeys = [o.key for o in allCosObjects if not o.key.endswith('.meta.json')]
        self._cosObjects = {
            o.key : CosObject(o.key, o.size) for o in allCosObjects
            if o.key in dataObjectKeys and o.key + '.meta.json' in metaObjectKeys
        }
        for o in set(dataObjectKeys).difference(self._cosObjects.keys()):
            logger.warning('Cos bucket contains no meta data for {}.'.format(o))
        totalSize = sum([o.size for o in self._cosObjects.values()])
        logger.info(
            'IBMCosBucket with {} uploadable objects, {} bytes total.'.format(
                len(self._cosObjects), totalSize
            )
        )
        
    def upload(self, fileName: Union[Path, str], objectName: str=None):
        '''
        Stores the local file `fileName` as well as `fileName.meta.json` in
        the bucket as `objectName` and `objectName.meta.json`. Will raise
        an exception of `fileName` or `fileName.meta.json` do not exist.
        
        :param fileName:   Name of (data) file to ingest.
        :param objectName: (Optional) name of file in bucket. `fileName`
                           without parent directories will be used if not
                           provided.
        '''
        try:
            fileName = Path(fileName)
            metaFileName = fileName.parent / (fileName.name + '.meta.json')
            if not fileName.is_file():
                raise OSError('File {} does not exist.'.format(fileName))
            if not metaFileName.is_file():
                raise OSError('File {} does not exist.'.format(metaFileName))
                
            if (objectName is None):
                objectName = fileName.name
            metaObjectName = objectName + '.meta.json'

            # Set 50 MB chunks
            partSize = 1024 * 1024 * 50

            # Set threshold to 50 MB
            fileThreshold = 1024 * 1024 * 50

            # Set the transfer threshold and chunk size
            transferConfig = ibm_boto3.s3.transfer.TransferConfig(
                multipart_threshold=fileThreshold,
                multipart_chunksize=partSize
            )

            # The upload_fileobj method will automatically execute a multi-part upload
            # in partSize chunks for all files over fileThreshold
            logger.debug('Pushing {} to COS.'.format(fileName.name))
            with open(fileName, "rb") as fp:
                self.cosResource.Object(self.bucketName, objectName).upload_fileobj(
                    Fileobj=fp,
                    Config=transferConfig
                )
            logger.debug('Pushing {} to COS.'.format(metaFileName.name))
            with open(metaFileName, "rb") as fp:
                self.cosResource.Object(self.bucketName, metaObjectName).upload_fileobj(
                    Fileobj=fp,
                    Config=transferConfig
                )
        except IBMClientError as e:
            logger.error("Client error: {}".format(e))
            raise
        except Exception as e:
            logger.error("Unable to complete multi-part upload: {}".format(e))
            raise
        else:
            self._cosObjects[objectName] = CosObject(objectName, fileName.lstat().st_size)
    
    def getMetaData(self, objectName: str) -> Dict:
        '''
        Loads the meta data for the object `objectName` from the bucket.
        '''
        if objectName not in self._cosObjects:
            logger.error('Cannot load PAIRS meta data from COS.')
            raise ValueError(
                'Object might not be in the bucket or lack a meta file. Run \'self.updateCosObjects\' and check \'self.cosObjects\' to see if the object is available.'
            )
        metaObjectName = objectName + '.meta.json'
        metaObject = self.cosHMACClient.get_object(Bucket=self.bucketName, Key=metaObjectName)
        metaDataAsString = metaObject['Body'].read().decode("utf-8")
        return json.loads(metaDataAsString)
    
    def generatePresignedURL(self, objectName: str, expirationTime: int=3600*24) -> str:
        '''
        Generates a presigned URL for the object `objectName`. The URL can
        then be handed to the PAIRS upload API.
        
        :param objectName:     Name of object in the bucket.
        :param expirationTime: Expiration time of the URL (in seconds).
        '''
        if objectName not in self._cosObjects:
            logger.error('Cannot generate presigned URL.')
            raise ValueError(
                'Object might not be in the bucket or lack a meta file. Run \'self.updateCosObjects\' and check \'self.cosObjects\' to see if the object is available.'
            )
        if self.cosHMACClient is None:
            logger.error('Cannot generate presigned URL.')
            raise Exception('Presigned URLs require HMAC authentication.')

        return self.cosHMACClient.generate_presigned_url(
            'get_object', Params={'Bucket': self.bucketName,'Key': objectName},
            ExpiresIn=expirationTime
        )
    
    def delete(self, objectName: str):
        '''
        Deletes a (data) object as well as the associated meta data
        `objectName.meta.json` from the bucket. Deletion will only be
        performed if the object is stored in the internal dictionary
        returned by cosObjects(). After deletion, the object will be
        removed from said dictionary.
        
        :param objectName: Name of (data) object to be deleted
        '''
        if objectName not in self._cosObjects:
            logger.error('Cannot delete object from COS.')
            raise ValueError(
                'Object might not be in the bucket or lack a meta file. Run \'self.updateCosObjects\' and check \'self.cosObjects\' to see if the object is available.'
            )
            
        metaObjectName = objectName + '.meta.json'
        logger.debug('Deleting {} and {}.'.format(objectName, metaObjectName))
        self.cosHMACClient.delete_objects(
            Bucket=self.bucketName,
            Delete={'Objects' : [{'Key' : objectName}, {'Key' : metaObjectName}]}
        )
        self._cosObjects.pop(objectName)
    
    @staticmethod
    def createIBMCosResource(apikey, resourceInstanceID, authEndpoint, endpoint):
        # Create resource
        try:
            cos = ibm_boto3.resource("s3",
                ibm_api_key_id=apikey,
                ibm_service_instance_id=resourceInstanceID,
                ibm_auth_endpoint=authEndpoint,
                config=IBMConfig(signature_version="oauth"),
                endpoint_url=endpoint
            )
        except IBMClientError as e:
            logger.errog('Cannot create COS resource. Exception: {}'.format(e))
            raise
        return cos
    
    @staticmethod
    def getHMACClient(accessKeyID, secretAccessKey, resourceInstanceID, authEndpoint, endpoint):
        return ibm_boto3.client(
            "s3", aws_access_key_id=accessKeyID, aws_secret_access_key=secretAccessKey,
            ibm_service_instance_id=resourceInstanceID,ibm_auth_endpoint=authEndpoint,
            config=IBMConfig(signature_version="s3v4"), endpoint_url=endpoint
        )

class SingleUpload:
    '''
    Simple data class that wraps a single upload.
    '''
    def __init__(
        self, status: str, localFile: Path, remoteObject: str, size: int,
        uploadInfo: Dict, exception: str
    ):
        self.status = status
        self.localFile = localFile
        self.remoteObject = remoteObject
        self.size = size
        self.uploadInfo = uploadInfo
        self.exception = exception

    def __repr__(self):
        return 'Status = ' + str(self.status) + \
               ' localFile = ' + str(self.localFile) + \
               ' remoteObject = ' + str(self.remoteObject) + \
               ' size = ' + str(self.size) +  \
               ' uploadInfo = ' + str(self.uploadInfo) +  \
               ' exception = ' + str(self.exception)
        
class PAIRSUpload:
    '''
    PAIRS Upload enables uploads to IBM PAIRS via the upload API.
    In general, the object should be constructed via one of the factory methods
    PAIRSUpload.fromCos and PAIRSUpload.fromLocal for upload from COS
    or from a local source respectively.
    
    In both cases, the presene of a COS bucket (cosBucket) as staging area
    for the bucket is required.
    
    After initializing the object, call run(nWorkers) to start the upload.
    
    Consult the internal dictionaries pendingUploads and completedUploads
    to learn about pending and completed uploads. However, do not modify
    these dictionaries.
    
    The arguments maxFiles and maxDataVolume define maximal number of files
    and maximal data volume to be sent to PAIRS. Violation of these bounds
    will currently only trigger a warning. run() can still be triggered.
    
    In general, each upload requires a data file `fileName` as well as an
    associated meta data file `fileName.meta.json` in the same location --
    whether COS or local path. This .meta.json file contains a JSON object
    specifying how the data in `fileName` is to be ingested. Consult the
    PAIRS upload API documentation regarding the details of this format.
    
    Note however that the key `url` should not appear in the meta file. This
    URL will be dynamically generated during ingestion.
    '''
    
    def __init__(
        self, cosBucket: CosBucket, pairsAuthHeader: Dict,
        deleteFromBucket: bool=False, pairsHost: str='https://pairs.res.ibm.com',
        ssl: Any=None, maxFiles: int=1024, maxDataVolume: float=1.0E10,
        uploads: List[SingleUpload]=list()
    ):
        self.cosBucket = cosBucket
        self.deleteFromBucket = deleteFromBucket
        
        self.pairsAuthHeader = pairsAuthHeader
        self.pairsHost = pairsHost
        self.ssl = ssl
        
        self.maxFiles = maxFiles
        self.maxDataVolume = maxDataVolume
        self.urlExpirationTime = 3600 * 24

        self.pendingUploads = uploads
        self.completedUploads = list()
        
        count = len(self.pendingUploads)
        size = sum([u.size for u in uploads if u.size])
        self._volumeString = '\'PAIRSUpload\'. Calling \'run\' will upload {} files. Total raw data volume {} bytes. Depending on the PAIRS resolution level of target layer(s), the data volume might be significantly different in IBM PAIRS.'.format(count, size)
        
        if count > maxFiles or size > maxDataVolume:
            warnings.warn(self._volumeString)
        else:
            logger.info(self._volumeString)

    
    def __repr__(self):
        return self._volumeString
    
    
    @classmethod
    def fromCos(
        cls, cosBucket: CosBucket, pairsAuthHeader: Dict, 
        deleteFromBucket: bool=False, pairsHost: str='https://pairs.res.ibm.com',
        ssl: Any=None, maxFiles: int=1024, maxDataVolume: float=1.0E10, **kwargs
    ) -> 'PAIRSUpload':
        '''
        Factory method used to initialize a PAIRSUpload object to
        ingest data into IBM PAIRS. Use this method to load data that is
        already present in the COS bucket `cosBucket`. The keyword arguments
        **kwargs will be handed to the cosBucket.updateCosObjects method and
        thus define which objects can be uploaded. For IBM COS, the
        `Prefix` argument is particularly useful (and should suffice for most
        cases). Note that for each object to be ingested, a associated meta
        data object with `objectName.meta.json` should also be in the same bucket.
        
        Usage example:
            uploadFromCos = uploads.PAIRSUpload.fromCos(
                cosBucket, pairsAuthHeader, Prefix='Prefix of objects to ingest.'
            )
            uploadFromCos.run(nWorkers=2)
        '''
        cosBucket.updateCosObjects(**kwargs)            
        
        cosObjects = cosBucket.cosObjects()
        uploads = [
            SingleUpload(
                status='COS', localFile=None, remoteObject=o.key,
                size=o.size, uploadInfo=dict(), exception=None
            )
            for o in cosObjects.values()
        ]

        return cls(
            cosBucket, pairsAuthHeader, deleteFromBucket, pairsHost, ssl,
            maxFiles, maxDataVolume, uploads
        )
        
    @classmethod
    def fromLocal(
        cls, cosBucket: CosBucket, pairsAuthHeader: Dict,
        deleteFromBucket: bool=False, pairsHost: str='https://pairs.res.ibm.com',
        ssl: Any=None, maxFiles: int=1024, maxDataVolume: float=1.0E10,
        fileList: List[Union[Path, str]]=list()
    ) -> 'PAIRSUpload':
        '''
        Factory method used to initialize a PAIRSUpload object to
        ingest data into IBM PAIRS. Use this method to load data that is
        located at a local source. The list `fileList` should contain
        the paths to data files (as strings or Path objects) that are to be ingested
        into IBM PAIRS. Note that for each file to be ingested, a associated meta
        data file with `fileName.meta.json` should also be in the same bucket.
        Data files with suffix `.meta.json` will thus be excluded, since they are
        assumed to contain meta data.
        
        Usage example:
            uploadFromLocal = uploads.PAIRSUpload.fromLocal(
                cosBucket, pairsAuthHeader, fileList=[myData.tiff]
            )
            uploadFromCos.run(nWorkers=1)
        '''
        uploads = list()
        
        for file in fileList:
            file = Path(file)
            if file.suffixes[-2:] == ['.meta', '.json']:
                continue
            metaFile = file.parent / (file.name + '.meta.json')
            if not file.is_file():
                raise ValueError('File {} does not seem to exist.'.format(f.name))
            if not metaFile.is_file():
                raise ValueError('Meta file {} does not seem to exist.'.format(metaFile.name))
            uploads.append(
                SingleUpload(
                    status='LOCAL',
                    localFile=file,
                    remoteObject=None,
                    size=file.lstat().st_size,
                    uploadInfo=dict(),
                    exception=None
                )
            )
                     
        return cls(
            cosBucket, pairsAuthHeader, deleteFromBucket, pairsHost, ssl,
            maxFiles, maxDataVolume, uploads
        )
    
    async def uploadWorker(self, session: aiohttp.ClientSession):
        '''
        Internal method.
        
        Keeps initializing uploads until self.pendingUploads is
        empty.
        '''
        def intify(i):
            try:
                return int(i)
            except ValueError:
                return i
        
        while self.pendingUploads:
            u = self.pendingUploads.pop()
            logger.debug('Worker picked up {}.'.format(u.localFile))

            if u.status == 'LOCAL':
                try:
                    self.cosBucket.upload(u.localFile)
                except Exception as e:
                    excDetails = ''.join(traceback.format_exception(*sys.exc_info()))
                    u.exception = excDetails
                    u.status = 'FAILED'
                    logger.error('Exception {} during COS upload.'.format(e))
                else:
                    u.remoteObject = u.localFile.name
                    u.status = 'COS'
                    logger.debug('Worker moved {} to COS.'.format(u.remoteObject))

            if u.status == 'COS':
                try:
                    logger.debug('Worker will obtain meta data for {}.'.format(u.remoteObject))
                    metaInfo = self.cosBucket.getMetaData(u.remoteObject)
                    metaInfo['url'] = self.cosBucket.generatePresignedURL(u.remoteObject)
                    metaInfo['datalayer_id'] = [intify(i) for i in metaInfo['datalayer_id']]
                    logger.debug('Worker has obtained meta data for {}.'.format(u.remoteObject))
                    uploadSucceeded, uploadStatus = await runPairsUpload(
                        session, metaInfo, statusInterval=20, timeout=60*60,
                        pairsHost=self.pairsHost
                    )
                    u.uploadInfo = uploadStatus
                except Exception as e:
                    logger.error('Exception {} during PAIRS upload.'.format(e))
                    excDetails = ''.join(traceback.format_exception(*sys.exc_info()))
                    u.exception = excDetails
                    u.status = 'FAILED'
                else:
                    if uploadSucceeded:
                        u.status = 'SUCCEEDED'
                        logger.debug('Upload of {} succeded.'.format(u.remoteObject))
                    else:
                        u.status = 'FAILED'
                        logger.debug('Upload of {} failed.'.format(u.remoteObject))
                    if uploadSucceeded and self.deleteFromBucket:
                        logger.debug('Worker will remove {} from COS.'.format(u.remoteObject))
                        self.cosBucket.delete(u.remoteObject)
                        logger.debug('Worker removed {} from COS.'.format(u.remoteObject))

            if u.status == 'SUCCEEDED':
                logger.debug('Upload succeded {}.'.format(u.remoteObject))
            elif u.status == 'FAILED':
                logger.warning('Upload failed {}.'.format(u.remoteObject))
            else:
                raise Exception('Worker completes processing in ill-defined status: {}.'.format(u.status))
            self.completedUploads.append(u)
            await asyncio.sleep(1)
    
    async def run(self, nWorkers: int=1):
        '''
        Triggers upload to IBM PAIRS. Up to nWorkers will be initializing and
        monitoring uploads in parallel.
        '''
        if nWorkers > MAX_WORKERS:
            raise ValueError('Maximum number of workers is {}.'.format(MAX_WORKERS))
        logger.debug('Commencing upload run.')
        
        connector = aiohttp.TCPConnector(ssl=self.ssl)
        async with aiohttp.ClientSession(connector=connector, headers=self.pairsAuthHeader) as session:
            logger.debug('Initialized ClientSession.')
            try:
                runningUploads = asyncio.gather(
                    *[self.uploadWorker(session) for i in range(nWorkers)],
                    return_exceptions=False
                )
                await runningUploads
            except Exception as e:
                logger.debug('Query workers triggered exception {}.'.format(e))
                raise
            else:
                logger.debug('All workers completed processing.')

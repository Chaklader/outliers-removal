import gzip

from pylogger import PyLogger
log=PyLogger.getLogger("dcpipeline")

def decompress_gzipped(data: bytes) -> bytes:
    try:
        return gzip.decompress(data)
    except Exception as error:
        log.error("Error decompressing gzipped data:", error)
        raise

def compress_gzipped(data: bytes) -> bytes:
    try:
        return gzip.compress(data)
    except Exception as error:
        log.error("Error compressing gzipped data:", error)
        raise 
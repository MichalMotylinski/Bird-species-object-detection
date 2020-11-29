import os
import enum
import weakref
from pathlib import Path


ERROR = {
    'SERVER_CONNECTION': {
        'code': 0,
        'message': '[ERROR] Error in server'
    },
    'DUP_FILE': {
        'code': 2,
        'message': '[ERROR] There are image files with same names'
    },
    'UNDEF_FILE': {
        'code': 3,
        'message': '[ERROR] There are image files with unavailable names'
    },
    'MAKE_DIR': {
        'NG_PATH': {
            'code': 110,
            'message': '[ERROR] No such a directory. chose others. '
        },
        'NG_USERNAME': {
            'code': 113,
            'message': '[ERROR] The username is unavailable. use only halfwidth-alphanumeric (0-9, a-z, A-Z) and under-bar (_).'
        }
    },
    'DELETION': {
        'XML': {
            'code': 120,
            'message': '[ERROR] Xml deletion failed!'
        },
        'PNG': {
            'code': 130,
            'message': '[ERROR] Png deletion failed!'
        }
    }

}

IMG_STATUS = {
    'LOADING': {
        'code': 105,
        'message': 'Loading images...'
    },
    'NO_IMG': {
        'code': 100,
        'message': 'No images found.'
    }
}

NOTICE = {
    'MAKE_DIR': {
        'INITIAL': {
            'code': 115,
            'message': 'create "public" directories'
        },
        'SUCCESS': {
            'code': 111,
            'message': 'Successfully created directories!'
        }
    },
    'DELETION': {
        'XML': {
            'SUCCESS': {
                'code': 121,
                'message': 'Png deletion successful!'
            }
        },
        'PNG': {
            'SUCCESS': {
                'code': 131,
                'message': 'Png deletion successful!'
            }
        }
    }
}


class Task(enum.Enum):
    CLASSIFICATION = 0
    DETECTION = 1
    SEGMENTATION = 2

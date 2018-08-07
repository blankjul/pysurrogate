import os

import matlab


class MatlabEngine:
    __instance = None

    @staticmethod
    def get_instance():
        if MatlabEngine.__instance is None:
            MatlabEngine.__instance = matlab.engine.start_matlab()

            dace_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '..', 'vendor', 'matlab', 'dace'))
            MatlabEngine.__instance.addpath(dace_path, nargout=0)

        return MatlabEngine.__instance


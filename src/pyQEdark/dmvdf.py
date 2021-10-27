"""
Basic class for describing a velocity distribution function.

author: Aria Radick
date: 7/19/21
"""

import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import interp1d
from os import path as osp
import sqlite3
from pyQEdark.constants import ckms, ccms, c_light

class DM_Halo:

    def __init__(self, save_loc=None, in_unit='kms', out_unit='kms', **kwargs):

        self.save_loc = save_loc

        if self.save_loc is not None:
            def adapt_array(arr):
                out = io.BytesIO()
                np.save(out, arr)
                out.seek(0)
                return sqlite3.Binary(out.read())

            def convert_array(text):
                out = io.BytesIO(text)
                out.seek(0)
                return np.load(out)

            sqlite3.register_adapter(np.ndarray, adapt_array)
            sqlite3.register_converter("array", convert_array)

            if not osp.exists(self.save_loc):
                self._init_db()

        self.allowed_keys = {'interp'}

        corr_dict = { 'kms' : ckms,
                      'cms' : ccms,
                      'ms'  : c_light,
                      'nat' : 1 }

        self.in_vcorr = corr_dict[in_unit]
        self.out_vcorr = corr_dict[out_unit]
        self.vcorr = corr_dict[out_unit] / corr_dict[in_unit]

        self.interp = True

        self.set_params(**kwargs)

    def _setup(self, **kwargs):
        from pyQEdark.vdfs import f_SHM, f_Tsa, f_MSW
        from pyQEdark.etas import etaSHM, etaTsa, etaMSW, etaFromVDF

        name_dict = { 'shm' : 'Standard Halo Model',
                      'tsa' : 'Tsallis Model',
                      'msw' : 'Empirical Model' }

        f_dict = { 'shm' : f_SHM,
                   'tsa' : f_Tsa,
                   'msw' : f_MSW }

        eta_dict = { 'shm' : etaSHM,
                     'tsa' : etaTsa,
                     'msw' : etaMSW }

        if 'vdf' in kwargs.keys():

            if isinstance(kwargs['vdf'], str):
                vdf = kwargs['vdf']
                self.vdf = vdf
                if 'vparams' in kwargs.keys():
                    self.vparams = kwargs['vparams']

                self.vname = name_dict[vdf]

                if vdf == 'msw':
                    fparams = [self.vparams[0], self.vparams[2],
                               self.vparams[3]]
                else:
                    fparams = [self.vparams[0], self.vparams[2]]

                f_VDF = f_dict[vdf](fparams, vp=0)
                self.f_VDF = lambda v: f_VDF(v) / self.vcorr**3

                v2f = f_dict[vdf](fparams, vp=2)
                self.v2f = lambda v: v2f(v) / self.vcorr

                self.v0 = self.vparams[0] / self.in_vcorr * ckms
                self.vE = self.vparams[1] / self.in_vcorr * ckms
                self.vesc = self.vparams[2] / self.in_vcorr * ckms
                if vdf == 'msw':
                    self.p = self.vparams[3]
                else:
                    self.p = None

                eta_fn = eta_dict[vdf](self.vparams)
                self.eta_fn = lambda x: eta_fn(x) / self.vcorr

                if vdf == 'shm' or not self.interp:
                    self.etaInterp = None
                    self.eta = self.eta_fn
                else:
                    self.etaInterp = self.make_etaInterp()
                    self.eta = self.etaInterp

            elif callable(kwargs['vdf']):
                self.f_VDF = lambda x: kwargs['vdf'](x) / self.vcorr**3
                self.v2f = lambda x: x**2 * self.f_VDF(x) / self.vcorr

                self.v0 = None
                self.vE = 232/ckms * self.in_vcorr
                self.vesc = 544/ckms * self.in_vcorr
                self.p = None

                self._set_custom_params(**kwargs)

                vesctmp = self.vesc
                vEtmp = self.vE

                self.vE *= ckms / self.in_vcorr
                self.vesc *= ckms / self.in_vcorr

                eta_fn = etaFromVDF(kwargs['vdf'], vesc=vesctmp, vE=vEtmp)
                self.eta_fn = lambda x: eta_fn(x) / self.vcorr

                if self.interp:
                    self.etaInterp = self.make_etaInterp()
                    self.eta = self.etaInterp
                else:
                    self.etaInterp = None
                    self.eta = self.eta_fn

            else:
                raise TypeError("Keyword argument 'vdf' must be a string " +\
                                "or function.")

        if 'eta' in kwargs.keys():

            if callable(kwargs['eta']):
                self.eta_fn = lambda x: kwargs['eta'](x) / self.vcorr

                self.v0 = None
                self.vE = 232/ckms * self.in_vcorr
                self.vesc = 544/ckms * self.in_vcorr
                self.p = None

                self._set_custom_params(**kwargs)

                self.vE *= ckms / self.in_vcorr
                self.vesc *= ckms / self.in_vcorr

                if self.interp:
                    self.etaInterp = self.make_etaInterp()
                    self.eta = self.etaInterp
                else:
                    self.etaInterp = None
                    self.eta = self.eta_fn

            else:
                raise TypeError("Keyword argument 'eta' must be a function.")

    def set_params(self, **kwargs):
        self.__dict__.update((k,v) for k,v in kwargs.items() \
                             if k in self.allowed_keys)

        if 'vdf' in kwargs.keys() or 'eta' in kwargs.keys():
            self._setup(**kwargs)

        elif 'vparams' in kwargs.keys():
            from utils.velocity_dists import f_SHM, f_Tsa, f_MSW
            from utils.etas import etaSHM, etaTsa, etaMSW

            name_dict = { 'shm' : 'Standard Halo Model',
                          'tsa' : 'Tsallis Model',
                          'msw' : 'Empirical Model' }

            f_dict = { 'shm' : f_SHM,
                       'tsa' : f_Tsa,
                       'msw' : f_MSW }

            eta_dict = { 'shm' : etaSHM,
                         'tsa' : etaTsa,
                         'msw' : etaMSW }

            self.vparams = kwargs['vparams']
            vdf = self.vdf

            if vdf == 'msw':
                fparams = [self.vparams[0], self.vparams[2],
                           self.vparams[3]]
            else:
                fparams = [self.vparams[0], self.vparams[2]]

            f_VDF = f_dict[vdf](fparams, vp=0)
            self.f_VDF = lambda v: f_VDF(v) / self.vcorr**3

            v2f = f_dict[vdf](fparams, vp=2)
            self.v2f = lambda v: v2f(v) / self.vcorr

            self.v0 = self.vparams[0] / self.in_vcorr * ckms
            self.vE = self.vparams[1] / self.in_vcorr * ckms
            self.vesc = self.vparams[2] / self.in_vcorr * ckms
            if vdf == 'msw':
                self.p = self.vparams[3]
            else:
                self.p = None

            eta_fn = eta_dict[vdf](self.vparams)
            self.eta_fn = lambda x: eta_fn(x) / self.vcorr

            if vdf == 'shm' or not self.interp:
                self.etaInterp = None
                self.eta = self.eta_fn
            else:
                self.etaInterp = self.make_etaInterp()
                self.eta = self.etaInterp

        elif 'interp' in kwargs.keys():
            if self.vdf == 'shm' or not self.interp:
                self.etaInterp = None
                self.eta = self.eta_fn
            else:
                self.etaInterp = self.make_etaInterp()
                self.eta = self.etaInterp

    def _set_custom_params(self, **kwargs):
        custom_keys = {'vE', 'vesc'}
        self.__dict__.update((k,v) for k,v in kwargs.items() \
                             if k in custom_keys)

    def _init_db(self):

        path_to_db = self.save_loc

        create_tables = """
        DROP TABLE IF EXISTS vdfs;
        CREATE TABLE vdfs (
            id INTEGER,
            name VARCHAR,
            v0 REAL,
            vE REAL,
            vesc REAL,
            p REAL,
            vmin_interp array,
            eta_interp array,
            PRIMARY KEY (id) );
        """

        conn = sqlite3.connect(path_to_db, detect_types=sqlite3.PARSE_DECLTYPES)
        cur = conn.cursor()
        cur.executescript(create_tables)
        conn.commit()
        conn.close()

    def _get_vdf_id(self):
        db_path = self.save_loc

        def insert_vdf(vars, path_to_db):

            shm_command = ''' INSERT INTO vdfs (name, v0, vE, vesc)
                              VALUES (?, ?, ?, ?) '''

            msw_command = ''' INSERT INTO vdfs (name, v0, vE, vesc, p)
                              VALUES (?, ?, ?, ?, ?) '''

            custom_command = ''' INSERT INTO vdfs (name, vE, vesc) VALUES
                                 (?, ?, ?) '''

            if vars[0] == 'shm' or vars[0] == 'tsa':
                command = shm_command

            elif vars[0] == 'msw':
                command = msw_command

            else:
                command = custom_command

            conn = sqlite3.connect(path_to_db,
                                   detect_types=sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor()
            cur.execute(command, vars)
            conn.commit()
            conn.close()

            return

        def find_vdf_id(vars, path_to_db):

            shm_command = '''SELECT id FROM vdfs WHERE name = ? AND v0 = ?
                             AND vE = ? AND vesc = ?'''

            msw_command = '''SELECT id FROM vdfs WHERE name = ? AND v0 = ?
                             AND vE = ? AND vesc = ? AND p = ?'''

            custom_command = '''SELECT id FROM vdfs WHERE name = ? AND vE = ?
                                AND vesc = ?'''

            if vars[0] == 'shm' or vars[0] == 'tsa':
                command = shm_command

            elif vars[0] == 'msw':
                command = msw_command

            else:
                command = custom_command

            conn = sqlite3.connect(path_to_db,
                                   detect_types=sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor()
            cur.execute(command, vars)
            f_ = cur.fetchone()
            conn.close()

            if f_ is not None:
                return f_[0]
            else:
                return f_

        if self.vdf == 'shm' or self.vdf == 'tsa':
            vdf = self.vdf
            v0 = np.around(float(self.v0), decimals=4)
            vE = np.around(float(self.vE), decimals=4)
            vesc = np.around(float(self.vesc), decimals=4)
            _vars = (vdf, v0, vE, vesc)

        elif self.vdf == 'msw':
            vdf = self.vdf
            v0 = np.around(float(self.v0), decimals=4)
            vE = np.around(float(self.vE), decimals=4)
            vesc = np.around(float(self.vesc), decimals=4)
            p = np.around(float(self.p), decimals=4)
            _vars = (vdf, v0, vE, vesc, p)

        else:
            vdf = self.vsavename
            vE = np.around(float(self.vE), decimals=4)
            vesc = np.around(float(self.vesc), decimals=4)
            _vars = (vdf, vE, vesc)

        tmp = find_vdf_id(_vars, db_path)
        if tmp is not None:
            return tmp

        else:
            insert_vdf(_vars, db_path)
            return find_vdf_id(_vars, db_path)

    def _get_eta_vals(self):
        db_path = self.save_loc

        vdf_id = int(self._get_vdf_id())

        def find_eta(id, path_to_db):
            command = '''SELECT vmin_interp, eta_interp FROM vdfs WHERE
                         id = ?'''

            conn = sqlite3.connect(path_to_db,
                                   detect_types=sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor()
            cur.execute( command, (id,) )
            f_ = cur.fetchone()
            conn.close()

            if f_[0] is not None:
                return f_
            else:
                return f_[0]

        def insert_eta(vars, path_to_db):
            command = ''' UPDATE vdfs SET vmin_interp = ?, eta_interp = ?
                          WHERE id = ? '''

            conn = sqlite3.connect(path_to_db,
                                   detect_types=sqlite3.PARSE_DECLTYPES)
            cur = conn.cursor()
            cur.execute(command, vars)
            conn.commit()
            conn.close()

            return

        tmp = find_eta(vdf_id, db_path)
        if tmp is not None:
            return tmp

        else:
            N_vmin = 10000
            Vmin = np.linspace(0, (self.vE+self.vesc+1)/ckms, N_vmin)
            Eta = self.eta_fn(Vmin*self.in_vcorr)*self.out_vcorr
            insert_eta((Vmin, Eta, vdf_id), db_path)
            return (Vmin, Eta)

    def make_etaInterp(self):
        if self.save_loc is not None:
            vmin, eta = self._get_eta_vals()

        else:
            N_vmin = 10000
            vmin = np.linspace(0, (self.vE+self.vesc+1)/ckms, N_vmin)
            eta = self.eta_fn(vmin*self.in_vcorr)*self.out_vcorr

        return interp1d(vmin*self.in_vcorr, eta/self.out_vcorr,
                        bounds_error=False, fill_value=0.)

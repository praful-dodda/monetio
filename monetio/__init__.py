from . import grids, models, obs, profile, sat
import xarray as xr

# point observations
airnow = obs.airnow
aeronet = obs.aeronet
aqs = obs.aqs
cems = obs.cems
crn = obs.crn
improve = obs.improve
ish = obs.ish
ish_lite = obs.ish_lite
nadp = obs.nadp
openaq = obs.openaq

# models
fv3chem = models.fv3chem
cmaq = models.cmaq
camx = models.camx
prepchem = models.prepchem
ncep_grib = models.ncep_grib
# emitimes = models.emitimes
# cdump2netcdf = models.cdump2netcdf
hysplit = models.hysplit
hytraj = models.hytraj
pardump = models.pardump

# profiles
icartt = profile.icartt
tolnet = profile.tolnet

# sat
goes = sat.goes

__all__ = ["models", "obs", "sat", "util", "grids", "profile"]


def wrap_longitudes(lons):
    """Short summary.

    Parameters
    ----------
    lons : type
        Description of parameter `lons`.

    Returns
    -------
    type
        Description of returned object.

    """
    return (lons + 180) % 360 - 180


def _rename_latlon(ds):
    """Short summary.

    Parameters
    ----------
    ds : type
        Description of parameter `ds`.

    Returns
    -------
    type
        Description of returned object.

    """
    if "latitude" in ds.coords:
        return ds.rename({"latitude": "lat", "longitude": "lon"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "lat", "Longitude": "lon"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "lat", "Lon": "lon"})
    else:
        return ds


def _monet_to_latlon(da):
    if isinstance(da, xr.DataArray):
        dset = da.to_dataset()
    dset["x"] = da.longitude[0, :].values
    dset["y"] = da.latitude[:, 0].values
    dset = dset.drop(["latitude", "longitude"])
    dset = dset.set_coords(["x", "y"])
    dset = dset.rename({"x": "lon", "y": "lat"})
    if isinstance(da, xr.DataArray):
        return dset[da.name]
    else:
        return dset


def _dataset_to_monet(dset, lat_name="latitude", lon_name="longitude", latlon2d=False):
    """Renames XArray DataArray or Dataset for use with monet functions

    Parameters
    ----------
    dset : xr.DataArray or xr.Dataset
        a given data obj to be renamed for monet.
    lat_name : str
        name of the latitude array.
    lon_name : str
        name of the longitude array.
    latlon2d : bool
        flag for if the latitude and longitude data is two dimensional.

    Returns
    -------
    type
        Description of returned object.

    """
    if "grid_xt" in dset.dims:
        # GFS v16 file
        try:
            if isinstance(dset, xr.DataArray):
                dset = _dataarray_coards_to_netcdf(dset, lat_name="grid_yt", lon_name="grid_xt")
            elif isinstance(dset, xr.Dataset):
                dset = _dataarray_coards_to_netcdf(dset, lat_name="grid_yt", lon_name="grid_xt")
            else:
                raise ValueError
        except ValueError:
            print("dset must be an Xarray.DataArray or Xarray.Dataset")

    if "south_north" in dset.dims:  # WRF WPS file
        dset = dset.rename(dict(south_north="y", west_east="x"))
        try:
            if isinstance(dset, xr.Dataset):
                if "XLAT_M" in dset.data_vars:
                    dset["XLAT_M"] = dset.XLAT_M.squeeze()
                    dset["XLONG_M"] = dset.XLONG_M.squeeze()
                    dset = dset.set_coords(["XLAT_M", "XLONG_M"])
                elif "XLAT" in dset.data_vars:
                    dset["XLAT"] = dset.XLAT.squeeze()
                    dset["XLONG"] = dset.XLONG.squeeze()
                    dset = dset.set_coords(["XLAT", "XLONG"])
            elif isinstance(dset, xr.DataArray):
                if "XLAT_M" in dset.coords:
                    dset["XLAT_M"] = dset.XLAT_M.squeeze()
                    dset["XLONG_M"] = dset.XLONG_M.squeeze()
                elif "XLAT" in dset.coords:
                    dset["XLAT"] = dset.XLAT.squeeze()
                    dset["XLONG"] = dset.XLONG.squeeze()
            else:
                raise ValueError
        except ValueError:
            print("dset must be an Xarray.DataArray or Xarray.Dataset")

    dset = _rename_to_monet_latlon(dset)
    latlon2d = True
    # print(len(dset[lat_name].shape))
    # print(dset)
    if len(dset[lat_name].shape) < 2:
        # print(dset[lat_name].shape)
        latlon2d = False
    if latlon2d is False:
        try:
            if isinstance(dset, xr.DataArray):
                dset = _dataarray_coards_to_netcdf(dset, lat_name=lat_name, lon_name=lon_name)
            elif isinstance(dset, xr.Dataset):
                dset = _coards_to_netcdf(dset, lat_name=lat_name, lon_name=lon_name)
            else:
                raise ValueError
        except ValueError:
            print("dset must be an Xarray.DataArray or Xarray.Dataset")
    else:
        dset = _rename_to_monet_latlon(dset)
    dset["longitude"] = wrap_longitudes(dset["longitude"])
    return dset


def _rename_to_monet_latlon(ds):
    """Short summary.

    Parameters
    ----------
    ds : type
        Description of parameter `ds`.

    Returns
    -------
    type
        Description of returned object.

    """
    if "lat" in ds.coords:
        return ds.rename({"lat": "latitude", "lon": "longitude"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "latitude", "Longitude": "longitude"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "latitude", "Lon": "longitude"})
    elif "XLAT_M" in ds.coords:
        return ds.rename({"XLAT_M": "latitude", "XLONG_M": "longitude"})
    elif "XLAT" in ds.coords:
        return ds.rename({"XLAT": "latitude", "XLONG": "longitude"})
    else:
        return ds


def _coards_to_netcdf(dset, lat_name="lat", lon_name="lon"):
    """Short summary.

    Parameters
    ----------
    dset : type
        Description of parameter `dset`.
    lat_name : type
        Description of parameter `lat_name`.
    lon_name : type
        Description of parameter `lon_name`.

    Returns
    -------
    type
        Description of returned object.

    """
    from numpy import meshgrid, arange

    lon = wrap_longitudes(dset[lon_name])
    lat = dset[lat_name]
    lons, lats = meshgrid(lon, lat)
    x = arange(len(lon))
    y = arange(len(lat))
    dset = dset.rename({lon_name: "x", lat_name: "y"})
    dset.coords["longitude"] = (("y", "x"), lons)
    dset.coords["latitude"] = (("y", "x"), lats)
    dset["x"] = x
    dset["y"] = y
    dset = dset.set_coords(["latitude", "longitude"])
    return dset


def _dataarray_coards_to_netcdf(dset, lat_name="lat", lon_name="lon"):
    """Short summary.

    Parameters
    ----------
    dset : type
        Description of parameter `dset`.
    lat_name : type
        Description of parameter `lat_name`.
    lon_name : type
        Description of parameter `lon_name`.

    Returns
    -------
    type
        Description of returned object.

    """
    from numpy import meshgrid, arange

    lon = wrap_longitudes(dset[lon_name])
    lat = dset[lat_name]
    lons, lats = meshgrid(lon, lat)
    x = arange(len(lon))
    y = arange(len(lat))
    dset = dset.rename({lon_name: "x", lat_name: "y"})
    dset.coords["latitude"] = (("y", "x"), lats)
    dset.coords["longitude"] = (("y", "x"), lons)
    dset["x"] = x
    dset["y"] = y
    return dset


def rename_latlon(ds):
    """Short summary.

    Parameters
    ----------
    ds : type
        Description of parameter `ds`.

    Returns
    -------
    type
        Description of returned object.

    """
    if "latitude" in ds.coords:
        return ds.rename({"latitude": "lat", "longitude": "lon"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "lat", "Longitude": "lon"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "lat", "Lon": "lon"})
    else:
        return ds


def rename_to_monet_latlon(ds):
    """Short summary.

    Parameters
    ----------
    ds : type
        Description of parameter `ds`.

    Returns
    -------
    type
        Description of returned object.

    """
    if "lat" in ds.coords:
        return ds.rename({"lat": "latitude", "lon": "longitude"})
    elif "Latitude" in ds.coords:
        return ds.rename({"Latitude": "latitude", "Longitude": "longitude"})
    elif "Lat" in ds.coords:
        return ds.rename({"Lat": "latitude", "Lon": "longitude"})
    elif "grid_lat" in ds.coords:
        return ds.rename({"grid_lat": "latitude", "grid_lon": "longitude"})
    else:
        return ds


def dataset_to_monet(dset, lat_name="lat", lon_name="lon", latlon2d=False):
    if len(dset[lat_name].shape) != 2:
        latlon2d = False
    if latlon2d is False:
        dset = coards_to_netcdf(dset, lat_name=lat_name, lon_name=lon_name)
    return dset


def coards_to_netcdf(dset, lat_name="lat", lon_name="lon"):
    from numpy import meshgrid, arange

    lon = dset[lon_name]
    lat = dset[lat_name]
    lons, lats = meshgrid(lon, lat)
    x = arange(len(lon))
    y = arange(len(lat))
    dset = dset.rename({lon_name: "x", lat_name: "y"})
    dset.coords["longitude"] = (("y", "x"), lons)
    dset.coords["latitude"] = (("y", "x"), lats)
    dset["x"] = x
    dset["y"] = y
    dset = dset.set_coords(["latitude", "longitude"])
    return dset


@xr.register_dataarray_accessor("monetio")
class MONETIOAccessor(object):
    """Short summary.

    Parameters
    ----------
    xray_obj : type
        Description of parameter `xray_obj`.

    Attributes
    ----------
    obj : type
        Description of attribute `obj`.

    """

    def __init__(self, xray_obj):
        """Short summary.

        Parameters
        ----------
        xray_obj : type
            Description of parameter `xray_obj`.

        Returns
        -------
        type
            Description of returned object.

        """
        self._obj = xray_obj

    def to_met3d_nc(self, filename=None):
        try:
            if filename is None:
                raise ValueError
        except ValueError:
            print("filename is None. Please provide filename to use this function")

        da = self._obj.copy()
        da = _dataset_to_monet(da)
        # add meta data to lat lon
        da.latitude.attrs["long_name"] = "latitude"
        da.latitude.attrs["standard_name"] = "latitude"
        da.latitude.attrs["units"] = "degrees_north"
        da.latitude.attrs["axis"] = "Y"
        da.longitude.attrs["long_name"] = "longitude"
        da.longitude.attrs["standard_name"] = "longitude"
        da.longitude.attrs["units"] = "degrees_east"
        da.longitude.attrs["axis"] = "X"
        # fillna
        da = da.fillna(-999e9)
        # fix time to hours since start date
        sdate = da.time.to_index().min()
        t = int((da.time.to_index() - sdate) / 3600)
        da["time"][:] = t
        da.time.attrs["calendar"] = "propleptic_gregorian"
        da.time.attrs["standard_name"] = "time"
        da.time.attrs["long_name"] = "time"
        da.time.attrs["units"] = sdate.strftime("hours since %Y-%m-%d %H:%M:%S")
        da.time.attrs["axis"] = "T"
        timeVar.standard_name = "time"
        timeVar.long_name = "time"
        timeVar.units = startdate.strftime("hours since %Y-%m-%d %H:%M:%S")
        timeVar.axis = "T"

        da.to_netcdf(filename)

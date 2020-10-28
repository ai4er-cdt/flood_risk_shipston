import requests
from typing import List


class NrfaDataCollector(object):
    def __init__(self):
        """
        A quick wrapper around the NRFA API to query data from their servers.
        """
        self.BASE_URL = "https://nrfaapps.ceh.ac.uk/nrfa/ws/"
        self._available_stations = self._get_available_stations()

    @property
    def available_stations(self) -> List[int]:
        return self._available_stations

    def _assert_valid_id(self, station_id: int) -> None:
        """
        Asserts that a given station id is valid

        Args:
            station_id (int): The station id to check
        """
        assert (
            station_id in self.available_stations
        ), f"Station id {station_id} does not exist."

    def _get_available_stations(self) -> List[int]:
        """
        Collects the list of all valid station ids from the NRFA API.
        """
        response = requests.get(f"{self.BASE_URL}station-ids?format=json-object")
        return response.json()["station-ids"]

    def get_static_data(self, station_id: int, fields: List[str]) -> dict:
        """
        API Call to request static data about a specific station from NRFA.

        Args:
            station_id (int): The ID of the station about which to request data
                (Shipsotn: 54106)
            fields (List[str]): The fields of data to request about the given station.
                Must be one of the available fields listed below or visible here:
                http://nrfaapps.ceh.ac.uk/nrfa/nrfa-api.html

        Available fields:
            id
                The station identifier.
            name
                The station name.
            default
                Both the id and name fields.
            catchment-area
                The catchment area (in km2).
            grid-reference
                The station grid reference. For JSON output the grid-reference is
                represented as an object with the following properties:
                ngr
                    (String) The grid reference in string form (i.e. SS9360201602).
                easting
                    (Number) The grid reference easting (in metres).
                northing
                    (Number) The grid reference northing (in metres).
            lat-long
                The station latitude/longitude. For JSON output the lat-long is
                represented as an object with the following properties:
                string
                    (String) The textual representation of the lat/long
                    (i.e. 50°48'15.0265"N 3°30'40.7121"W).
                latitude
                    (Number) The latitude (expressed in decimal degrees).
                longitude
                    (Number) The longitude (expressed in decimal degrees).
            river
                The name of the river.
            location
                The name of the location on the river.
            station-level
                The altitude of the station, in metres, above Ordnance Datum or,
                in Northern Ireland, Malin Head.
            easting
                The grid reference easting.
            northing
                The grid reference northing.
            station-information
                Basic station information: id, name, catchment-area, grid-reference,
                lat-long, river, location, station-level, measuring-authority-id,
                measuring-authority-station-id, hydrometric-area, opened, closed,
                station-type, bankfull-flow, structurefull-flow, sensitivity.
            category
                Information about the main station categories: nrfa-mean-flow,
                nrfa-peak-flow, feh-pooling, feh-qmed, feh-neither, nhmp, benchmark,
                live-data.
            catchment-information
                Basic catchment information: factors-affecting-runoff.
            gdf-statistics
                Gauged daily flow statistics: gdf-start-date, gdf-end-date,
                gdf-mean-flow, gdf-min-flow, gdf-first-date-of-min,
                gdf-last-date-of-min, gdf-max-flow, gdf-first-date-of-max,
                gdf-last-date-of-max, gdf-q95-flow, gdf-q70-flow, gdf-q50-flow,
                gdf-q10-flow, gdf-q05-flow, gdf-base-flow-index, gdf-day-count,
                gdf-flow-count.
            peak-flow-statistics
                Basic peak-flow statistics: peak-flow-start-date, peak-flow-end-date,
                 qmed.
            elevation
                Catchment elevation pecentile data: minimum-altitude,
                10-percentile-altitude, 50-percentile-altitude, 90-percentile-altitude,
                maximum-altitude.
            catchment-rainfall
                Catchment rainfall standard period data: saar-1941-1970, saar-1961-1990.
            lcm2000
                Land cover map data (2000): lcm2000-woodland,
                lcm2000-arable-horticultural, lcm2000-grassland,
                lcm2000-mountain-heath-bog, lcm2000-urban.
            lcm2007
                Land cover map data (2007): lcm2007-woodland,
                lcm2007-arable-horticultural, lcm2007-grassland,
                lcm2007-mountain-heath-bog, lcm2007-urban.
            geology
                Catchment geology data: high-perm-bedrock, moderate-perm-bedrock,
                low-perm-bedrock, mixed-perm-bedrock, high-perm-superficial,
                low-perm-superficial, mixed-perm-superficial.
            feh-descriptors
                FEH catchment descriptors: propwet, bfihost, farl, dpsbar.
            urban-extent
                Urban extent data: urbext-1990, urbext-2000.
            spatial-location
                The grid reference and lat/long as individual fields: easting,
                northing, latitude, longitude.
            peak-flow-metadata
                Metadata related to peak-flow data: peak-flow-rejected-amax-years,
                peak-flow-rejected-periods.
            all
                Everything.

        Returns:
            (dict): The response json parsed as python dict.
        """
        self._assert_valid_id(station_id)

        # Assemble API call URL
        url = (
            f"{self.BASE_URL}station-info?station={station_id}"
            f"&fields={','.join(fields)}"
            f"&format=json-object"
        )
        # Get response from API
        response = requests.get(url)

        # Return if successful response, raise exception otherwise
        if int(response.status_code) == 200:
            return response.json()
        else:
            failure_info = response.content.decode().split("<p>")[-1].split("</p>")[0]
            raise Exception(
                f"Request failed with status code: {response.status_code} and error message {failure_info}"
            )

    def get_timeseries(self, station_id: int, data_type: str) -> dict:
        """
        Requests time series data for a given station.

        Args:
            station_id (int): The ID of the station about which to request data
                (Shipsotn: 54106)
            data_type (str): The time-series data to request. Must be one of the
                available data_types listed below or here:
                http://nrfaapps.ceh.ac.uk/nrfa/nrfa-api.html#parameter-data-type

        Available data_types:
            gdf
                Gauged daily flows
            ndf
                Naturalised daily flows
            gmf
                Gauged monthly flows
            nmf
                Naturalised monthly flows
            cdr
                Catchment daily rainfall
            cdr-d
                Catchment daily rainfall distance to rain gauge
            cmr
                Catchment monthly rainfall
            pot-stage
                Peaks over threshold stage
            pot-flow
                Peaks over threshold flow
            gauging-stage
                Gauging stage
            gauging-flow
                Gauging flow
            amax-stage
                Annual maxima stage
            amax-flow
                Annual maxima flow

        Returns:
            dict: The returned json response parsed as dict.
        """
        self._assert_valid_id(station_id)

        url = (
            f"{self.BASE_URL}time-series?station={station_id}"
            f"&data-type={data_type}"
            f"&format=json-object"
        )
        # Get response from API
        response = requests.get(url)

        # Return if successful response, raise exception otherwise
        if int(response.status_code) == 200:
            return response.json()
        else:
            failure_info = response.content.decode().split("<p>")[-1].split("</p>")[0]
            raise Exception(
                f"Request failed with status code: {response.status_code} and error message {failure_info}"
            )

import datetime

def pretty_print_output(output):
    """
    Pretty-print the travel output dictionary in a well-structured, 
    presentation-friendly format with tables and embedded links.
    """
    
    # ---------------------------------------------------------------------
    # 0. Utility Functions
    # ---------------------------------------------------------------------
    
    def ascii_table(headers, rows, title=None):
        """
        Generate an ASCII table from headers and rows.
        headers: list of column header strings
        rows: list of lists, each inner list corresponds to one row of data
        title: optional string to print above the table
        
        Returns a single string containing the formatted table.
        """
        if not headers:
            return ""
        
        # Convert all row cells to string
        str_rows = []
        for r in rows:
            str_rows.append([str(item) if item is not None else "" for item in r])
        
        # Compute column widths
        col_widths = [len(h) for h in headers]
        for r in str_rows:
            for i, cell in enumerate(r):
                col_widths[i] = max(col_widths[i], len(cell))
        
        # Build the horizontal divider
        def build_divider(col_widths):
            line_elems = ["+" + "-"*(w+2) for w in col_widths]
            return "".join(line_elems) + "+"
        
        divider = build_divider(col_widths)
        
        # Build the header line
        def build_header(headers, col_widths):
            line = ""
            for h, w in zip(headers, col_widths):
                line += "| " + h + " "*(w - len(h)) + " "
            line += "|"
            return line
        
        header_line = build_header(headers, col_widths)
        
        # Build each row line
        def build_row(row, col_widths):
            line = ""
            for cell, w in zip(row, col_widths):
                line += "| " + cell + " "*(w - len(cell)) + " "
            line += "|"
            return line
        
        # Construct the final lines
        lines = []
        if title:
            lines.append(title)
        lines.append(divider)
        lines.append(header_line)
        lines.append(divider)
        for r in str_rows:
            lines.append(build_row(r, col_widths))
        lines.append(divider)
        
        return "\n".join(lines)
    
    def format_duration(minutes):
        """
        Given total minutes, return a string like '4h36m'.
        """
        if not isinstance(minutes, int):
            return str(minutes)
        hrs = minutes // 60
        mins = minutes % 60
        return f"{hrs}h {mins}m"
    
    def embed_link(text, url):
        """
        Return a short embedded link style: [text](url)
        If url is None or empty, return plain text.
        """
        if url:
            return f"[{text}]({url})"
        return text
    
    def extract_numeric_price(p):
        """
        Convert a price string like '$299' into a float: 299.0
        If parsing fails, return a large fallback so sorting doesn't break.
        """
        if not p:
            return 9999999
        try:
            # strip out '$' or any non-digit/decimal
            return float(p.replace('$', '').strip())
        except:
            return 9999999
    
    def parse_event_datetime(evt):
        """
        Parse the event's Date and Time into a datetime object for sorting.
        If either is invalid, return a 'max' datetime so it sorts last.
        """
        date_str = evt.get('Date', '')
        time_str = evt.get('Time', '00:00:00')
        if not date_str:
            return datetime.datetime.max
        
        dt_str = f"{date_str} {time_str}"
        try:
            return datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.datetime.max
        

    # ---------------------------------------------------------------------
    # A. Trip Overview
    # ---------------------------------------------------------------------
    print("\n======================== TRIP OVERVIEW ======================\n")
    # We'll create a small table to display basic trip info if available.
    trip_headers = ["Field", "Value"]
    trip_rows = []
    
    # Pull each field from output if it exists, otherwise default to empty.
    location_val    = output.get('location', "")
    destination_val = output.get('destination', "")
    
    # Handle start_date / end_date which might be date objects
    start_date_val  = output.get('start_date', "")
    end_date_val    = output.get('end_date', "")
    # Convert date objects to string if needed
    if isinstance(start_date_val, datetime.date):
        start_date_val = start_date_val.isoformat()  # e.g. "2025-02-15"
    if isinstance(end_date_val, datetime.date):
        end_date_val = end_date_val.isoformat()
    
    budget_val      = output.get('budget', "")
    adults_val      = output.get('num_adults', "")
    children_val    = output.get('num_children', "")
    
    trip_rows.append(["Location", location_val])
    trip_rows.append(["Destination", destination_val])
    trip_rows.append(["Start Date", start_date_val])
    trip_rows.append(["End Date", end_date_val])
    trip_rows.append(["Budget", budget_val])
    trip_rows.append(["Adults", adults_val])
    trip_rows.append(["Children", children_val])
    
    overview_table = ascii_table(trip_headers, trip_rows, title="TRIP OVERVIEW")
    print(overview_table)
    print()
    
    # ---------------------------------------------------------------------
    # 1. Flights
    # ---------------------------------------------------------------------

    
    if 'flights' in output and output['flights']:
        print("\n========================== FLIGHTS ==========================\n")
        
        # Check if the flight data is from the new node (has 'departure flights' and 'arrival flights')
        is_new_node = isinstance(output['flights'], list) and len(output['flights']) > 0 and 'departure flights' in output['flights'][0]
        
        # Define the max number of flights to display per class
        max_flights_per_class = 10
        
        if is_new_node:
            # Handle the new node structure
            for flight_set in output['flights']:
                def print_flight_table(title, flights):
                    """
                    Prints a formatted table for a given flight category.
                    """
                    if not flights:
                        return
                    
                    # Organize flights by travel class
                    categorized_flights = {"Economy": [], "Business": [], "First": []}
                    
                    for flight in flights:
                        travel_class = flight.get('travel_class', 'Economy')  # Default to Economy
                        categorized_flights.setdefault(travel_class, []).append(flight)
                    
                    # Print separate tables for each class
                    for class_name, class_flights in categorized_flights.items():
                        if class_flights:
                            print(f"\n--- {title} ({class_name} Class) ---\n")
                            rows = []
                            for i, flight in enumerate(class_flights[:max_flights_per_class], start=1):
                                airline = flight.get('airline', '')
                                departure_time = flight.get('departure_time', '')
                                arrival_time = flight.get('arrival_time', '')
                                departure_airport = flight.get('departure_airport', '')
                                arrival_airport = flight.get('arrival_airport', '')
                                duration = flight.get('duration', '')
                                stops = flight.get('stops', '')
                                price = flight.get('price', '')
                                booking_link = embed_link("Book", flight.get('booking_url', '')) if flight.get('booking_url') else "N/A"
                                
                                rows.append([
                                    str(i),
                                    airline,
                                    departure_time,
                                    arrival_time,
                                    f"{departure_airport} -> {arrival_airport}",
                                    duration,
                                    str(stops),
                                    price,
                                    booking_link
                                ])
                            
                            # Build the table
                            headers = ["#", "Airline", "Departure", "Arrival", "Route", "Duration", "Stops", "Price", "Booking"]
                            table_str = ascii_table(headers, rows)
                            print(table_str)

                # Print Departure Flights
                if 'departure flights' in flight_set and flight_set['departure flights']:
                    print_flight_table("DEPARTURE FLIGHTS", flight_set['departure flights'])

                # Print Arrival Flights
                if 'arrival flights' in flight_set and flight_set['arrival flights']:
                    print_flight_table("ARRIVAL FLIGHTS", flight_set['arrival flights'])
        else:
            # Handle the old node structure
            for idx, flight_set in enumerate(output['flights']):
                print(f"Flight Option #{idx+1}\n")
                
                # The classes we might have
                travel_classes = ["Economy", "Premium Economy", "Business", "First"]
                
                for tclass in travel_classes:
                    if tclass not in flight_set:
                        continue  # skip if that class doesn't exist in the data
                    
                    class_data = flight_set[tclass]
                    best = class_data.get('best_flights', [])
                    other = class_data.get('other_flights', [])
                    
                    # Decide which list to use
                    # If best is non-empty, use best. Otherwise use other.
                    relevant_flights = best if best else other
                    if not relevant_flights:
                        continue
                    
                    # Sort by price ascending
                    relevant_flights.sort(key=lambda f: f.get('price', 9999999))
                    
                    # Take up to 3
                    relevant_flights = relevant_flights[:5]
                    
                    # Prepare table rows
                    rows = []
                    for i, fdata in enumerate(relevant_flights, start=1):
                        airlines_str = ", ".join(fdata.get('airlines', []))
                        price_str = f"${fdata.get('price','')}"
                        route_str = f"{fdata.get('departure_airport','')} -> {fdata.get('arrival_airport','')}"
                        times_str = f"{fdata.get('departure_time','')} -> {fdata.get('arrival_time','')}"
                        duration_str = format_duration(fdata.get('total_duration', ''))
                        
                        layovers = fdata.get('layovers', [])
                        layover_count = len(layovers)
                        
                        carbon_str = f"{fdata.get('carbon_emissions', '')} kg" if fdata.get('carbon_emissions') else ""
                        
                        # For an actual booking link, you might have a real URL or token. 
                        # If 'booking_token' is not None, you could generate a link. Example:
                        # booking_link = your_link_builder_function(fdata['booking_token'])
                        # For now, we'll just store the token if it exists.
                        token = fdata.get('booking_token', None)
                        if token:
                            booking_str = embed_link("Book Here", f"{token}")
                        else:
                            booking_str = "N/A"
                        
                        rows.append([
                            str(i),
                            airlines_str,
                            price_str,
                            route_str,
                            times_str,
                            duration_str,
                            str(layover_count),
                            tclass,
                            booking_str
                        ])
                    
                    # Build the table
                    headers = ["#", "Airlines", "Price", "Route", "Times", 
                               "Duration", "Layovers", "Class", "Booking"]
                    table_str = ascii_table(headers, rows, title=f"--- {tclass.upper()} ---")
                    print(table_str)
                    print()
    
  
    # ---------------------------------------------------------------------
    # 2. Accommodation
    # ---------------------------------------------------------------------
    
    if 'accommodation' in output and output['accommodation']:
        print("\n====================== ACCOMMODATIONS ======================\n")
        
        # Sort by price if possible
        accommodations = sorted(output['accommodation'], key=lambda x: extract_numeric_price(x.get('price')))
        
        # Take only first 4
        accommodations = accommodations[:12]
        
        rows = []
        for i, ac in enumerate(accommodations, start=1):
            name = ac.get('name', '')
            price = ac.get('price', '')
            rating = ac.get('rating', '')
            link = ac.get('link', None)
            
            link_str = embed_link("Click Here", link) if link else "N/A"
            
            rows.append([
                str(i),
                name,
                price,
                rating,
                link_str
            ])
        
        headers = ["#", "Name", "Price", "Rating", "Booking Link"]
        table_str = ascii_table(headers, rows, title="ACCOMMODATION OPTIONS")
        print(table_str)
        print()
    
    # ---------------------------------------------------------------------
    # 3. Activities
    # ---------------------------------------------------------------------
    
    if 'activities' in output and output['activities']:
        print("\n======================== ACTIVITIES ========================\n")
        
        rows = []
        for i, act in enumerate(output['activities'], start=1):
            name = act.get('name', '')
            address = act.get('address', '')
            description = act.get('description', '')
            
            # If address not found, fallback to description
            address_or_desc = address if address else description
            
            rows.append([
                str(i),
                name,
                address_or_desc
            ])
        
        headers = ["#", "Activity Name", "Address/Description"]
        table_str = ascii_table(headers, rows, title="THINGS TO DO")
        print(table_str)
        print()
    
    # ---------------------------------------------------------------------
    # 4. Live Events
    # ---------------------------------------------------------------------
    
    if 'live_events' in output and output['live_events']:
        print("\n======================== LIVE EVENTS =======================\n")

        # Sort events by parsed date/time
        live_events = sorted(output['live_events'], key=parse_event_datetime)

        def truncate_event_name(name, word_limit=7):
            """
            Truncates the event name to the first `word_limit` words.
            If the name is short, it remains unchanged.
            """
            words = name.split()
            if len(words) > word_limit:
                return " ".join(words[:word_limit]) + "..."
            return name

        rows = []
        for i, evt in enumerate(live_events, start=1):
            event_name = evt.get('Event', '')
            truncated_event_name = truncate_event_name(event_name)  # Truncate name

            date = evt.get('Date', '')
            time = evt.get('Time', '')
            venue = evt.get('Venue', '')
            url = evt.get('Url', None)

            link_str = embed_link("View Event", url) if url else "N/A"

            # Skip city/country as requested
            rows.append([
                str(i),
                truncated_event_name,
                date,
                time,
                venue,
                link_str
            ])

        headers = ["#", "Event", "Date", "Time", "Venue", "Link"]
        table_str = ascii_table(headers, rows, title="UPCOMING LIVE EVENTS")
        print(table_str)
        print()
        
    
    
    # ---------------------------------------------------------------------
    # 5. Recommendations
    # ---------------------------------------------------------------------
    
    if 'recommendations' in output and output['recommendations']:
        print("\n====================== RECOMMENDATIONS =====================\n")
        
        # We can build a table with columns: #, Category, Detail
        # Each item in 'recommendations' is a dict; we can flatten the key-value pairs.
        
        rows = []
        row_count = 0
        for rec_dict in output['recommendations']:
            # rec_dict is e.g. {"Crime Rate": "Brief info ...", "AnotherKey": "..."}
            for key, val in rec_dict.items():
                row_count += 1
                rows.append([
                    str(row_count),
                    str(key),
                    str(val)
                ])
        
        headers = ["#", "Category", "Information"]
        table_str = ascii_table(headers, rows, title="ADDITIONAL RECOMMENDATIONS")
        print(table_str)
        print()

    print("\n====================== END OF RESULTS ======================\n")



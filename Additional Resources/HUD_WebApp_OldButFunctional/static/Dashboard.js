// Make the datatable
                            dcc.Graph(
                                id='sec_chart',
                                figure=go.Figure(
                                    data=go.Pie(

                                        values=tweets['emotion'].value_counts().values,
                                        labels=tweets['emotion'].value_counts().index,
                                        textinfo='label',
                                        hole=.3
                                    ),
                                    layout=go.Layout(
                                        title_text='Secondary Category Breakdown (Hover for count)',
                                    )
                                )
                            )

// Make the first pie chart


var data = [{
    values: [16, 15, 12, 6, 5, 4, 42],
    labels: ['US', 'China', 'European Union', 'Russian Federation', 'Brazil', 'India', 'Rest of World'
    ],
    type: 'pie'
}, {
    values: [27, 11, 25, 8, 1, 3, 25],
    labels: ['US', 'China', 'European Union', 'Russian Federation', 'Brazil', 'India', 'Rest of World'
    ],
    text: 'CO2',
    textposition: 'inside',
    domain: {column: 1},
    name: 'CO2 Emissions',
    hoverinfo: 'label+percent+name',
    hole: .4,
    type: 'pie'
}];

var layout = {
    title: 'Global Emissions 1990-2011',
    annotations: [
        {
            font: {
                size: 20
            },
            showarrow: false,
            text: 'GHG',
            x: 0.17,
            y: 0.5
        },
        {
            font: {
                size: 20
            },
            showarrow: false,
            text: 'CO2',
            x: 0.82,
            y: 0.5
        }
    ],
    height: 400,
    width: 600,
    showlegend: false,
    grid: {rows: 1, columns: 2}
};
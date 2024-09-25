import React from 'react';
import { VictoryBar, VictoryChart, VictoryAxis } from 'victory';

const HorizontalBarChart = ({ data }) => {
  const d= {...data}
  delete d.arch
  const formattedData = Object.entries(d)
    .map(([label, value]) => ({ label, value:value*100 }))
    .sort((a, b) => b.value - a.value);

  return (
   <>
      <h2> Probability Bar Chart </h2>
      <p style={{marginBottom:0}}>It seems like your flower is <span style={{textTransform:"capitalize"}}> {formattedData[0].label}</span></p>
      <VictoryChart  horizontal>
        <VictoryAxis 
          style={{ tickLabels: { angle: 50, fontSize: 8, textTransform: "capitalize",fill:"#F1E9DB" } }}
          tickValues={formattedData.map(({ value }) => value)} 
        />
        <VictoryAxis dependentAxis 
        maxDomain={100}
        minDomain={0}
        style={{ tickLabels: { angle: 50, fontSize: 8, textTransform: "capitalize",fill:"#F1E9DB" } }}
        tickFormat={(tick) => tick.toFixed(2)} />
        <VictoryBar
          data={formattedData.reverse()} 
          barRatio={0.5}
          x="label"
          y="value"
          barWidth={10}
          style={{ data: { fill: '#6DAEDB' } }} 
        />
      </VictoryChart>
    </>
  );
};

export default HorizontalBarChart;

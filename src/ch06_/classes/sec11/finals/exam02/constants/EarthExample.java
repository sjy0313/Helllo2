package ch06_.classes.sec11.finals.exam02.constants;

public class EarthExample {
	public static void main(String[] args) {
		//상수 읽기
		System.out.println("지구의 반지름: " + Earth.EARTH_RADIUS + "km");
		System.out.println("지구의 표면적: " + Earth.EARTH_SURFACE_AREA + "km^2");
		System.out.printf("지구의 표면적 : %.7f Km^2\n", Earth.EARTH_SURFACE_AREA);
		System.out.printf("지구의 표면적 : %d Km^2\n", (int)Earth.EARTH_SURFACE_AREA);
	
	}
	
}
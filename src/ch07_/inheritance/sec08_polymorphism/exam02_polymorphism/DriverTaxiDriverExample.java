package ch07_.inheritance.sec08_polymorphism.exam02_polymorphism;

public class DriverTaxiDriverExample {
	public static void main(String[] args) {
	
		BusDriver busDriver = new BusDriver();
		TaxiDriver taxiDriver = new TaxiDriver();
		

		//매개값으로 Bus 객체를 제공하고 driver() 메소드 호출
		Bus bus = new Bus();
		busDriver.drive(bus);

		//매개값으로 Taxi 객체를 제공하고 driver() 메소드 호출
		Taxi taxi = new Taxi();
		taxiDriver.drive(taxi);
	}
}